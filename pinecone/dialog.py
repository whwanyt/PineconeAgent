"""对话控制器：把“唤醒 → 聆听 → 识别 → 处理 → 播报 → 追问”串起来，并实现对话打断。

这是整个音箱的“大脑皮层”：它订阅前端事件（唤醒、开始说话、说完），驱动状态机，
把识别出的文字交给 :class:`DialogHandler`（技能路由 / AI Agent），再把回复播报出来。

一轮对话的时序::

    WakeWordDetected ─▶ LISTENING（提示音，开 VAD，启动“没开口”超时）
    SpeechEnded      ─▶ THINKING（托管任务「对话」：识别 → handler.handle → 播报）
    播报开始          ─▶ SPEAKING（提高唤醒阈值；如开启 VAD 打断则保持 VAD）
    播报结束          ─▶ FOLLOW_UP（追问窗口，开 VAD）─超时─▶ IDLE
    追问窗口内开口    ─▶ LISTENING ...

打断（barge-in）：
    THINKING / SPEAKING 时
      - 检测到唤醒词（barge_in.wake_word）；或
      - VAD 检测到开口（barge_in.vad，且回声消除生效）
    → 取消所有前台托管任务（Agent 推理、技能、播报）→ 立即静音 → 进入 LISTENING。
    后台任务（音乐、长时间自定义任务）不受影响。

“轮次”（turn）：
    每次唤醒 / 打断都会让轮次号 +1。异步流程在每个 await 之后检查轮次号，
    如果已经不是自己的轮次（说明被打断了），就不再改变状态，避免旧流程把新流程的状态改乱。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from pinecone.audio.utils import beep
from pinecone.core.events import (
    BargeIn,
    SpeechEnded,
    SpeechStarted,
    StateChanged,
    Transcribed,
    WakeWordDetected,
)
from pinecone.core.state import AssistantState as S
from pinecone.core.task_manager import RunningTask, TaskKind

if TYPE_CHECKING:
    from pinecone.audio.output import AudioOutput
    from pinecone.core.bus import EventBus
    from pinecone.core.config import AppConfig
    from pinecone.core.state import StateMachine
    from pinecone.core.task_manager import TaskManager
    from pinecone.speech.asr.base import SpeechRecognizer
    from pinecone.speech.pipeline import SpeechPipeline
    from pinecone.speech.tts.speaker import Speaker

logger = logging.getLogger(__name__)


@dataclass
class Reply:
    """对一句用户指令的回复。

    ``text`` 与 ``stream`` 二选一：普通技能返回完整文本，AI Agent 返回流式文本。
    两者都为空表示不需要播报（例如“停止”指令）。
    """

    text: str | None = None
    stream: AsyncIterable[str] | None = None
    follow_up: bool = True
    """播报完是否进入追问窗口（AI 闲聊适合开启；“已停止”这类回复可以关闭）。"""


class DialogHandler(Protocol):
    """处理一句已识别文本的对象（技能路由器实现此协议）。"""

    async def handle(self, text: str, *, hint: str | None = None) -> Reply:
        """处理用户说的话并返回回复。运行在可取消的前台任务中。

        Args:
            text: 用户原话。
            hint: 技能转交时的情况说明，不播报，仅提供给兜底 AI。
        """
        ...


class EchoHandler:
    """调试用处理器：复述听到的话。"""

    async def handle(self, text: str, *, hint: str | None = None) -> Reply:
        return Reply(text=f"你说的是：{text}")


class DialogController:
    """对话控制器。只在事件循环线程中使用。"""

    def __init__(
        self,
        *,
        config: AppConfig,
        bus: EventBus,
        state: StateMachine,
        tasks: TaskManager,
        speaker: Speaker,
        output: AudioOutput,
        handler: DialogHandler,
        pipeline: SpeechPipeline | None = None,
        asr: SpeechRecognizer | None = None,
        aec_effective: bool = False,
    ) -> None:
        self._config = config
        self._bus = bus
        self._state = state
        self._tasks = tasks
        self._speaker = speaker
        self._output = output
        self._handler = handler
        self._pipeline = pipeline
        self._asr = asr
        self._turn = 0
        self._timer: asyncio.TimerHandle | None = None
        self._beep = beep(sample_rate=16000)

        barge = config.audio.barge_in
        # VAD 打断必须在回声消除生效时才启用，否则扬声器的声音会让音箱自己打断自己
        self._vad_barge_in = barge.vad and aec_effective
        if barge.vad and not aec_effective:
            logger.warning(
                "已配置 barge_in.vad，但回声消除未生效，VAD 打断已自动关闭（仍可用唤醒词打断）"
            )

    def start(self) -> None:
        """订阅事件，开始工作。"""
        self._bus.subscribe(WakeWordDetected, self._on_wake)
        self._bus.subscribe(SpeechStarted, self._on_speech_started)
        self._bus.subscribe(SpeechEnded, self._on_speech_ended)
        self._bus.subscribe(StateChanged, self._on_state_changed)

    def set_handler(self, handler: DialogHandler) -> None:
        """替换指令处理器（技能路由器初始化完成后调用）。"""
        self._handler = handler

    # ================================================================== 对外接口

    async def interrupt(self, reason: str = "user") -> int:
        """打断：取消所有前台任务并立即静音。返回被取消的任务数。

        在托管任务内调用时（例如“停止”技能），调用方自身所在的任务不会被取消。
        """
        cancelled = self._tasks.cancel_foreground(reason)
        self._speaker.stop()
        return cancelled

    async def speak(self, text: str) -> None:
        """主动播报（闹钟、自定义任务等使用），带状态管理。

        空闲 / 追问 / 播报中时会进入 SPEAKING 并在播完后回到 IDLE；
        如果用户正在说话或系统正在思考，则只播报不改变状态，避免打乱当前对话。
        可取消：被取消时立即停止播报。
        """
        if self._state.state in (S.IDLE, S.FOLLOW_UP, S.SPEAKING):
            turn = self._next_turn()
            await self._state.transition(S.SPEAKING, "主动播报")
            try:
                await self._speaker.say(text)
            finally:
                if turn == self._turn:
                    await self._state.transition(S.IDLE, "主动播报结束")
        else:
            await self._speaker.say(text)

    def announce(
        self, text: str, *, name: str = "播报", kind: TaskKind = TaskKind.FOREGROUND
    ) -> RunningTask:
        """在托管任务中主动播报（不等待播完），返回任务句柄。"""
        return self._tasks.spawn(self.speak(text), name=name, kind=kind, source="dialog")

    async def handle_text(self, text: str) -> None:
        """直接处理一句文本（跳过唤醒和识别），用于文本调试模式。"""
        turn = self._next_turn()
        if self._state.is_busy():
            await self.interrupt("barge_in")
        if self._state.state != S.LISTENING:
            await self._state.transition(S.LISTENING, "文本输入")
        await self._state.transition(S.THINKING, "文本输入")
        await self._bus.publish(Transcribed(text=text))
        rt = self._tasks.spawn(self._respond(text, turn), name="对话", source="dialog")
        await rt.wait()

    # ================================================================== 事件处理

    async def _on_wake(self, event: WakeWordDetected) -> None:
        if self._state.is_busy():
            if not self._config.audio.barge_in.wake_word:
                return
            logger.info("唤醒词打断")
            await self.interrupt("barge_in")
            await self._bus.publish(BargeIn(source="wake_word"))

        turn = self._next_turn()
        await self._state.transition(S.LISTENING, f"唤醒：{event.keyword}")
        self._arm_timer(self._config.vad.no_speech_timeout_s, lambda: self._on_listen_timeout(turn))

        dialog_cfg = self._config.dialog
        if dialog_cfg.wake_beep:
            self._tasks.spawn(self._output.play(self._beep, 16000), name="提示音", source="dialog")
        if dialog_cfg.wake_reply:
            self._tasks.spawn(
                self._speaker.say(dialog_cfg.wake_reply), name="唤醒应答", source="dialog"
            )

    async def _on_speech_started(self, _: SpeechStarted) -> None:
        state = self._state.state
        if state == S.FOLLOW_UP:
            self._cancel_timer()
            self._next_turn()
            await self._state.transition(S.LISTENING, "追问")
        elif state == S.LISTENING:
            self._cancel_timer()
        elif self._state.is_busy() and self._vad_barge_in:
            logger.info("开口打断")
            self._next_turn()
            await self.interrupt("barge_in")
            await self._bus.publish(BargeIn(source="vad"))
            await self._state.transition(S.LISTENING, "开口打断")

    async def _on_speech_ended(self, event: SpeechEnded) -> None:
        if self._state.state != S.LISTENING:
            return
        self._cancel_timer()
        turn = self._turn
        await self._state.transition(S.THINKING, "说完")
        self._tasks.spawn(self._process_audio(event.audio, turn), name="对话", source="dialog")

    async def _on_state_changed(self, event: StateChanged) -> None:
        """根据状态调整前端：VAD 开关、唤醒阈值。"""
        if self._pipeline is None:
            return
        new = S(event.new)
        boost = self._config.audio.barge_in.wake_threshold_boost if new == S.SPEAKING else 0.0
        self._pipeline.set_wake_threshold_boost(boost)

        if new == S.LISTENING:
            # 不重置：开口打断 / 追问时，用户的话已经开始了，重置会丢掉开头
            self._pipeline.set_vad_enabled(True, reset=False)
        elif new == S.FOLLOW_UP:
            self._pipeline.set_vad_enabled(True, reset=True)
        elif new == S.THINKING:
            self._pipeline.set_vad_enabled(self._vad_barge_in, reset=True)
        elif new == S.SPEAKING:
            self._pipeline.set_vad_enabled(self._vad_barge_in, reset=False)
        else:
            self._pipeline.set_vad_enabled(False)

    # ================================================================== 对话流程

    async def _process_audio(self, audio: np.ndarray, turn: int) -> None:
        """识别一句话并回复（运行在托管任务「对话」中）。"""
        assert self._asr is not None
        text = await self._asr.transcribe(audio)
        if turn != self._turn:
            return
        if not text:
            await self._state.transition(S.IDLE, "没听清")
            return
        await self._bus.publish(Transcribed(text=text))
        await self._respond(text, turn)

    async def _respond(self, text: str, turn: int) -> None:
        """调用处理器并播报回复。出错时播报兜底提示。"""
        try:
            reply = await self._handler.handle(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("处理指令出错：%s", text)
            reply = Reply(text="抱歉，我出了点问题，请稍后再试。", follow_up=False)
        if turn != self._turn:
            return

        if reply.text or reply.stream is not None:
            await self._state.transition(S.SPEAKING, "回复")
            if reply.stream is not None:
                await self._speaker.say_stream(reply.stream)
            else:
                await self._speaker.say(reply.text or "")
        if turn != self._turn:
            return

        follow_up_s = self._config.dialog.follow_up_s
        if reply.follow_up and follow_up_s > 0:
            await self._state.transition(S.FOLLOW_UP, "等待追问")
            self._arm_timer(follow_up_s, lambda: self._on_follow_up_timeout(turn))
        else:
            await self._state.transition(S.IDLE, "对话结束")

    # ================================================================== 超时

    async def _on_listen_timeout(self, turn: int) -> None:
        if turn == self._turn and self._state.state == S.LISTENING:
            await self._state.transition(S.IDLE, "没有说话")

    async def _on_follow_up_timeout(self, turn: int) -> None:
        if turn == self._turn and self._state.state == S.FOLLOW_UP:
            await self._state.transition(S.IDLE, "追问超时")

    def _arm_timer(self, delay: float, callback: Callable[[], Awaitable[None]]) -> None:
        """设置（替换）唯一的超时定时器。"""
        self._cancel_timer()
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(delay, lambda: loop.create_task(callback()))

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _next_turn(self) -> int:
        self._cancel_timer()
        self._turn += 1
        return self._turn
