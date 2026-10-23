"""对话控制器测试：完整一轮对话、追问超时、唤醒词打断、VAD 打断（使用假组件）。"""

import asyncio

import numpy as np

from pinecone.core.bus import EventBus
from pinecone.core.config import AppConfig
from pinecone.core.events import (
    BargeIn,
    SpeechEnded,
    SpeechStarted,
    StateChanged,
    WakeWordDetected,
)
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
    def __init__(self, results: list[str] | None = None) -> None:
        self.results = list(results or [])

    async def transcribe(self, audio) -> str:
        return self.results.pop(0) if self.results else "现在几点"


class Handler:
    async def handle(self, text: str, *, hint: str | None = None) -> Reply:
        return Reply(text=f"回复：{text}")


def build(
    *,
    speaker_delay: float = 0.0,
    vad_barge_in: bool = False,
    aec: bool = False,
    handler: object | None = None,
    asr_results: list[str] | None = None,
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
        asr=FakeAsr(asr_results),  # type: ignore[arg-type]
        aec_effective=aec,
    )
    dialog.start()
    return bus, state, tasks, speaker, pipeline, dialog


async def wait_state(state: StateMachine, target: S, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while state.state != target:  # noqa: ASYNC110 - 测试中简单轮询即可
            await asyncio.sleep(0.005)


async def test_full_turn_and_follow_up_timeout() -> None:
    """唤醒 → 聆听 → 思考 → 播报 → 追问窗口 → 超时回到空闲。"""
    bus, state, _, speaker, pipeline, _ = build()
    await bus.publish(WakeWordDetected(keyword="小派小派"))
    assert state.state == S.LISTENING and pipeline.vad_enabled

    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.FOLLOW_UP)
    assert speaker.said == ["回复：现在几点"]
    await wait_state(state, S.IDLE)
    assert not pipeline.vad_enabled


async def test_wake_word_barge_in_keeps_background_tasks() -> None:
    """播报时说唤醒词：播报被取消、进入聆听；后台任务不受影响。"""
    bus, state, tasks, speaker, pipeline, _ = build(speaker_delay=10)
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
        bus, state, tasks, _, _, _ = build(speaker_delay=10, vad_barge_in=True, aec=aec)
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

    bus, state, tasks, _, _, _ = build(handler=Asking())
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


async def _say(bus: EventBus) -> None:
    """模拟用户说了一句（开口 + 说完）。"""
    await bus.publish(SpeechStarted())
    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))


async def test_unheard_speech_keeps_listening() -> None:
    """识别结果为空时回到追问窗口，接着说的话照常处理，不必重新唤醒。"""
    bus, state, tasks, speaker, pipeline, _ = build(asr_results=[""])
    await bus.publish(WakeWordDetected(keyword="小派小派"))
    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.FOLLOW_UP)
    assert speaker.said == [] and pipeline.vad_enabled

    await _say(bus)
    await wait_state(state, S.FOLLOW_UP)
    assert speaker.said == ["回复：现在几点"]
    await tasks.shutdown()


async def test_repeated_noise_ends_the_dialog_and_expires() -> None:
    """连续没听清超过上限就结束对话，并调用回复的 on_expire。"""
    expired: list[int] = []

    class Asking:
        async def handle(self, text: str, *, hint: str | None = None) -> Reply:
            async def expire() -> None:
                expired.append(1)

            return Reply(text="你要哪一个？", listen_s=5, on_expire=expire)

    bus, state, tasks, _, _, _ = build(handler=Asking(), asr_results=["第二", "", "", ""])
    await bus.publish(WakeWordDetected(keyword="小派小派"))
    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.FOLLOW_UP)
    for _ in range(2):
        await _say(bus)
        await wait_state(state, S.FOLLOW_UP)
        assert not expired
    await _say(bus)
    await wait_state(state, S.IDLE)
    assert expired == [1]
    await tasks.shutdown()


def _record_states(bus: EventBus) -> list[str]:
    states: list[str] = []
    bus.subscribe(StateChanged, lambda e: states.append(e.new))
    return states


async def test_announcement_waits_for_the_reply() -> None:
    """回复播报中来了闹钟播报：等回复念完再念，状态不会在回复中途切回空闲。"""
    bus, state, tasks, speaker, _, dialog = build(speaker_delay=0.05)
    states = _record_states(bus)
    await bus.publish(WakeWordDetected(keyword="小派小派"))
    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.SPEAKING)

    alarm = tasks.spawn(dialog.speak("闹钟响了"), name="闹钟", kind=TaskKind.BACKGROUND)
    await alarm.wait()
    assert speaker.said == ["回复：现在几点", "闹钟响了"]
    assert states[-4:] == ["speaking", "follow_up", "speaking", "idle"]
    await tasks.shutdown()


async def test_speech_inside_the_turn_is_not_held_back() -> None:
    """这轮对话里的技能自己播报（例如前台自定义任务）：直接念，不等自己。"""
    holder: list[DialogController] = []

    class Speaking:
        async def handle(self, text: str, *, hint: str | None = None) -> Reply:
            await holder[0].speak("我的IP地址是一二三")
            return Reply(text="好了", follow_up=False)

    bus, state, tasks, speaker, _, dialog = build(handler=Speaking(), speaker_delay=0.01)
    holder.append(dialog)
    await bus.publish(WakeWordDetected(keyword="小派小派"))
    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.IDLE)
    assert speaker.said == ["我的IP地址是一二三", "好了"]
    await tasks.shutdown()


async def test_announcement_waits_while_the_user_is_talking() -> None:
    """用户正在说话时来了播报：等这轮回复念完再念，不在聆听时出声。"""
    bus, state, tasks, speaker, _, dialog = build()
    await bus.publish(WakeWordDetected(keyword="小派小派"))
    alarm = tasks.spawn(dialog.speak("闹钟响了"), name="闹钟", kind=TaskKind.BACKGROUND)
    await asyncio.sleep(0.02)
    assert speaker.said == [] and state.state == S.LISTENING

    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await alarm.wait()
    assert speaker.said == ["回复：现在几点", "闹钟响了"]
    await tasks.shutdown()


async def test_failed_turn_returns_to_idle() -> None:
    """识别或播放出错时复位到空闲，不卡在思考 / 播报中。"""

    class BrokenAsr:
        async def transcribe(self, audio) -> str:
            raise RuntimeError("识别模型崩了")

    class BrokenSpeaker(FakeSpeaker):
        async def say(self, text: str) -> str:
            raise RuntimeError("声卡掉线")

    for broken in ("asr", "speaker"):
        bus, state, tasks, _, _, dialog = build()
        if broken == "asr":
            dialog._asr = BrokenAsr()  # type: ignore[assignment]
        else:
            dialog._speaker = BrokenSpeaker()  # type: ignore[assignment]
        await bus.publish(WakeWordDetected(keyword="小派小派"))
        await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
        await wait_state(state, S.IDLE)
        await tasks.shutdown()


async def test_turn_cancelled_elsewhere_returns_to_idle() -> None:
    """「对话」任务被别处取消（不是打断，轮次没变）时也复位到空闲。"""
    bus, state, tasks, _, _, _ = build(speaker_delay=10)
    await bus.publish(WakeWordDetected(keyword="小派小派"))
    await bus.publish(SpeechEnded(audio=np.zeros(1600, np.float32)))
    await wait_state(state, S.SPEAKING)
    tasks.cancel_foreground("user")
    await wait_state(state, S.IDLE)
    await tasks.shutdown()
