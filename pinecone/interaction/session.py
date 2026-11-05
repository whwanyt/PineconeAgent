"""会话门面：交互层驱动核心的唯一入口。

在整体架构中的位置::

    TUI / 以后的 HTTP ──▶ Session ──▶ 对话控制器（文本与打断）
                                └──▶ Playback（播放列表，可选）

界面只依赖本模块的 :class:`Frontend` 和事件总线上的事件，不 import 对话控制器、
技能或 Agent。核心反过来也不依赖本包：由 :class:`pinecone.app.App` 在启动时把
已经建好的对话控制器和播放控制接进来。

``stop`` 只置一个事件，可以在信号处理函数里调用。其余方法都在事件循环线程上
``await``，会把取消继续传给对话控制器或播放控制。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from pinecone.core.bus import EventBus
from pinecone.core.events import PlaylistSnapshot
from pinecone.core.state import StateMachine
from pinecone.core.task_manager import TaskManager


class Playback(Protocol):
    """播放控制。音乐技能实现它，会话门面只按这个形状调用。

    选择曲目不经过意图路由，因此不会打断正在进行的对话，也不会写成一句用户消息。
    """

    def snapshot(self) -> PlaylistSnapshot:
        """当前队列。只读，不改播放位置。"""
        ...

    async def select(self, index: int) -> None:
        """从第 ``index`` 首开始放（从 1 数）。可取消。"""
        ...


class _Dialog(Protocol):
    """会话门面需要的对话控制器表面。由 :class:`~pinecone.dialog.DialogController` 实现。"""

    @property
    def timer_remaining(self) -> float | None:
        """聆听或追问还剩几秒，没有在计时时为 None。"""
        ...

    async def handle_text(self, text: str) -> None:
        """处理一句文本。可取消。"""
        ...

    async def interrupt(self, reason: str = "user") -> int:
        """打断当前一轮。可取消。"""
        ...


class Frontend(Protocol):
    """全屏界面需要的后端。测试用假对象，运行时是 :class:`Session`。"""

    bus: EventBus
    state: StateMachine
    tasks: TaskManager
    ready: asyncio.Event
    assistant_name: str
    data_path: Path

    @property
    def timer_remaining(self) -> float | None:
        """聆听或追问还剩几秒。"""
        ...

    async def run(self) -> None:
        """启动核心并阻塞到退出。"""
        ...

    def stop(self) -> None:
        """请求退出。"""
        ...

    async def submit_text(self, text: str) -> None:
        """提交一句文本。"""
        ...

    async def interrupt(self) -> None:
        """打断当前一轮。"""
        ...

    def playlist(self) -> PlaylistSnapshot:
        """当前播放队列。"""
        ...

    async def select_track(self, index: int) -> None:
        """从第 ``index`` 首开始放。"""
        ...


class Session:
    """把对话和播放收成一个对象，供各种交互方式共用。

    播放控制可以不接（没有音乐技能时）。这时 :meth:`playlist` 返回空快照，
    :meth:`select_track` 什么都不做。
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        state: StateMachine,
        tasks: TaskManager,
        ready: asyncio.Event,
        assistant_name: str,
        data_path: Path,
        run: Callable[..., Awaitable[None]],
        stop: Callable[[], None],
    ) -> None:
        self.bus = bus
        self.state = state
        self.tasks = tasks
        self.ready = ready
        self.assistant_name = assistant_name
        self.data_path = data_path
        self._run = run
        self._stop = stop
        self._dialog: _Dialog | None = None
        self._playback: Playback | None = None

    def attach(self, dialog: _Dialog, playback: Playback | None = None) -> None:
        """核心启动完成后接上对话控制器，以及可选的播放控制。

        Args:
            dialog: 已经 ``start`` 过的对话控制器。
            playback: 音乐技能；没有启用音乐时为 None。
        """
        self._dialog = dialog
        self._playback = playback

    @property
    def timer_remaining(self) -> float | None:
        """聆听或追问还剩几秒。对话控制器还没接上时为 None。"""
        if self._dialog is None:
            return None
        return self._dialog.timer_remaining

    async def submit_text(self, text: str) -> None:
        """把一句文本交给对话流程，和语音识别结果走同一条路。

        可取消：取消会传到对话控制器里正在处理的这一轮。
        只能在事件循环线程调用。

        Raises:
            RuntimeError: 对话控制器还没接上。
        """
        await self._require_dialog().handle_text(text)

    async def interrupt(self) -> None:
        """打断正在思考或播报的这一轮，后台任务不受影响。

        可取消。只能在事件循环线程调用。

        Raises:
            RuntimeError: 对话控制器还没接上。
        """
        await self._require_dialog().interrupt("user")

    def stop(self) -> None:
        """请求整个应用退出。

        只通知运行循环结束，本身不等待。可以从信号处理函数里调用。
        """
        self._stop()

    def playlist(self) -> PlaylistSnapshot:
        """当前播放队列的快照。没有接播放控制时为空。"""
        if self._playback is None:
            return PlaylistSnapshot()
        return self._playback.snapshot()

    async def select_track(self, index: int) -> None:
        """从第 ``index`` 首开始放（从 1 数）。

        不经过意图路由，所以不会打断对话，也不会记成一句用户话。
        没有接播放控制时什么都不做。可取消。只能在事件循环线程调用。
        """
        if self._playback is None:
            return
        await self._playback.select(index)

    async def run(self) -> None:
        """启动核心并阻塞到 :meth:`stop`。

        文本由 :meth:`submit_text` 进入，核心自己不读标准输入。
        """
        await self._run()

    def _require_dialog(self) -> _Dialog:
        if self._dialog is None:
            raise RuntimeError("会话还没准备好")
        return self._dialog
