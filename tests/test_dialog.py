"""对话控制器测试：完整一轮对话、追问超时、唤醒词打断、VAD 打断（使用假组件）。"""

import asyncio

import numpy as np

from pinecone.core.bus import EventBus
from pinecone.core.config import AppConfig
from pinecone.core.events import BargeIn, SpeechEnded, SpeechStarted, WakeWordDetected
from pinecone.core.state import AssistantState as S
from pinecone.core.state import StateMachine
from pinecone.core.task_manager import TaskKind, TaskManager
from pinecone.dialog import DialogController, Reply


class FakeSpeaker:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.said: list[str] = []
        self.stopped = 0

    async def say(self, text: str) -> str:
        await asyncio.sleep(self.delay)
        self.said.append(text)
        return text

    async def say_stream(self, pieces) -> str:
        text = "".join([p async for p in pieces])
        return await self.say(text)

    def stop(self) -> None:
        self.stopped += 1


class FakeOutput:
    async def play(self, samples, rate) -> None:
        return None


class FakePipeline:
    def __init__(self) -> None:
        self.vad_enabled = False
        self.boost = 0.0

    def set_vad_enabled(self, enabled: bool, *, reset: bool = True) -> None:
        self.vad_enabled = enabled

    def set_wake_threshold_boost(self, boost: float) -> None:
        self.boost = boost


class FakeAsr:
    async def transcribe(self, audio) -> str:
        return "现在几点"


class Handler:
    async def handle(self, text: str, *, hint: str | None = None) -> Reply:
        return Reply(text=f"回复：{text}")


def build(
    *,
    speaker_delay: float = 0.0,
    vad_barge_in: bool = False,
    aec: bool = False,
    handler: object | None = None,
):
    cfg = AppConfig()
    cfg.dialog.wake_beep = False
    cfg.dialog.follow_up_s = 0.1
    cfg.audio.barge_in.vad = vad_barge_in
    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())
    state = StateMachine(bus)
    tasks = TaskManager(bus)
    speaker = FakeSpeaker(speaker_delay)
    pipeline = FakePipeline()
    dialog = DialogController(
        config=cfg,
        bus=bus,
        state=state,
        tasks=tasks,
        speaker=speaker,  # type: ignore[arg-type]
        output=FakeOutput(),  # type: ignore[arg-type]
        handler=handler or Handler(),  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        asr=FakeAsr(),  # type: ignore[arg-type]
        aec_effective=aec,
    )
    dialog.start()
    return bus, state, tasks, speaker, pipeline


async def wait_state(state: StateMachine, target: S, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while state.state != target:  # noqa: ASYNC110 - 测试中简单轮询即可
            await asyncio.sleep(0.005)


async def test_full_turn_and_follow_up_timeout() -> None:
    """唤醒 → 聆听 → 思考 → 播报 → 追问窗口 → 超时回到空闲。"""
    bus, state, _, speaker, pipeline = build()
    await bus.publish(WakeWordDetected(keyword="小派小派"))
    assert state.state == S.LISTENING and pipeline.vad_enabled

    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.FOLLOW_UP)
    assert speaker.said == ["回复：现在几点"]
    await wait_state(state, S.IDLE)
    assert not pipeline.vad_enabled


async def test_wake_word_barge_in_keeps_background_tasks() -> None:
    """播报时说唤醒词：播报被取消、进入聆听；后台任务不受影响。"""
    bus, state, tasks, speaker, pipeline = build(speaker_delay=10)
    music = tasks.spawn(asyncio.sleep(10), name="音乐", kind=TaskKind.BACKGROUND)
    barge: list[BargeIn] = []
    bus.subscribe(BargeIn, barge.append)

    await bus.publish(WakeWordDetected(keyword="小派小派"))
    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.SPEAKING)
    assert pipeline.boost > 0  # 播报期间提高唤醒阈值

    await bus.publish(WakeWordDetected(keyword="小派小派"))
    assert state.state == S.LISTENING and barge[0].source == "wake_word"
    await asyncio.sleep(0.02)
    assert [t.name for t in tasks.list()] == ["音乐"] and not music.done
    assert speaker.said == [] and speaker.stopped == 1
    await tasks.shutdown()


async def test_vad_barge_in_requires_aec() -> None:
    """开口打断只在回声消除生效时启用。"""
    for aec, expected in ((False, S.SPEAKING), (True, S.LISTENING)):
        bus, state, tasks, _, _ = build(speaker_delay=10, vad_barge_in=True, aec=aec)
        await bus.publish(WakeWordDetected(keyword="小派小派"))
        await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
        await wait_state(state, S.SPEAKING)
        await bus.publish(SpeechStarted())
        assert state.state == expected
        await tasks.shutdown()


async def test_on_expire_runs_only_when_follow_up_times_out() -> None:
    """追问窗口超时才调用 on_expire；窗口内开口则不调用。"""
    expired: list[str] = []

    class Asking:
        async def handle(self, text: str, *, hint: str | None = None) -> Reply:
            async def expire() -> None:
                expired.append(text)

            return Reply(text="你要哪一个？", on_expire=expire)

    bus, state, tasks, _, _ = build(handler=Asking())
    await bus.publish(WakeWordDetected(keyword="小派小派"))
    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.FOLLOW_UP)
    await bus.publish(SpeechStarted())
    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.FOLLOW_UP)
    await wait_state(state, S.IDLE)
    await asyncio.sleep(0.01)
    assert expired == ["现在几点"]
    await tasks.shutdown()
