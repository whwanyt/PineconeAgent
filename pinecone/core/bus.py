"""异步事件总线。

各模块之间通过事件解耦：唤醒词模块只负责“发布检测到唤醒词”，
并不关心谁会响应；状态机、打断控制、日志等模块各自订阅自己关心的事件。

线程模型：
    - 总线本身运行在 asyncio 事件循环中；
    - 音频回调运行在 PortAudio 的线程里，**不能**直接 ``await``，
      必须使用 :meth:`EventBus.publish_threadsafe` 把事件投递回事件循环。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pinecone.core.events import Event

logger = logging.getLogger(__name__)

E = TypeVar("E", bound=Event)
# 处理函数既可以是普通函数，也可以是协程函数
Handler = Callable[[Any], Awaitable[None] | None]


class EventBus:
    """简单的发布/订阅事件总线。

    - 按事件类型分发，订阅父类会同时收到所有子类事件（例如订阅 :class:`Event` 可以收到全部事件）；
    - 处理函数按订阅顺序依次执行，单个处理函数抛异常只记录日志，不影响其他处理函数；
    - 不保证跨事件的处理顺序以外的任何事情，处理函数应尽量快速返回，
      耗时工作请交给 :class:`~pinecone.core.task_manager.TaskManager`。
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Handler]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """绑定事件循环，供 :meth:`publish_threadsafe` 使用。应用启动时调用一次。"""
        self._loop = loop

    def subscribe(self, event_type: type[E], handler: Callable[[E], Any]) -> Callable[[], None]:
        """订阅某类事件。

        Args:
            event_type: 事件类型（订阅父类可收到所有子类事件）。
            handler: 处理函数，普通函数或协程函数。

        Returns:
            取消订阅的函数，调用即可移除该处理函数。
        """
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

        return unsubscribe

    async def publish(self, event: Event) -> None:
        """发布事件，并等待所有处理函数执行完毕。

        可取消：如果发布者所在任务被取消，正在执行的处理函数会一起收到取消。
        """
        for handler in self._iter_handlers(type(event)):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "事件处理函数出错：%s 处理 %s", _name(handler), type(event).__name__
                )

    def publish_threadsafe(self, event: Event) -> None:
        """从非事件循环线程（例如音频回调线程）发布事件。

        线程安全：内部通过 ``call_soon_threadsafe`` 把发布动作调度到事件循环线程执行，
        调用方不会被阻塞，也拿不到处理结果。
        """
        if self._loop is None or self._loop.is_closed():
            logger.warning("事件总线尚未绑定事件循环，丢弃事件 %s", type(event).__name__)
            return
        self._loop.call_soon_threadsafe(lambda: self._loop.create_task(self.publish(event)))  # type: ignore[union-attr]

    async def wait_for(
        self,
        event_type: type[E],
        predicate: Callable[[E], bool] | None = None,
        timeout: float | None = None,
    ) -> E:
        """等待下一个满足条件的事件。

        Args:
            event_type: 事件类型。
            predicate: 可选过滤条件。
            timeout: 超时秒数，``None`` 表示一直等待。

        Returns:
            第一个满足条件的事件。

        Raises:
            TimeoutError: 超时。
        """
        future: asyncio.Future[E] = asyncio.get_running_loop().create_future()

        def _on_event(event: E) -> None:
            if not future.done() and (predicate is None or predicate(event)):
                future.set_result(event)

        unsubscribe = self.subscribe(event_type, _on_event)
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            unsubscribe()

    def _iter_handlers(self, event_type: type[Event]) -> list[Handler]:
        """按 MRO 收集处理函数：先具体类型，再父类型。复制列表，避免处理中途订阅变化。"""
        handlers: list[Handler] = []
        for cls in event_type.__mro__:
            if cls in self._handlers:
                handlers.extend(self._handlers[cls])  # type: ignore[index]
        return handlers


def _name(handler: Handler) -> str:
    return getattr(handler, "__qualname__", repr(handler))
