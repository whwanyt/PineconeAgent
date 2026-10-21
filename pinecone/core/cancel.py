"""取消令牌（CancelToken）。

为什么在 ``asyncio.Task.cancel()`` 之外还需要取消令牌？

1. **线程里的同步代码收不到 asyncio 的取消**：自定义任务的 python 动作、插件技能里
   可能有 ``time.sleep`` 或循环计算，它们跑在线程池中，``Task.cancel()`` 只能让
   ``await asyncio.to_thread(...)`` 那一层提前返回，线程本身仍会继续执行。
   这些代码需要在循环里主动检查 ``token.cancelled`` 或调用 ``token.raise_if_cancelled()``。
2. **需要知道取消原因**：例如“超时”“用户语音取消”“被打断”，``on_cancel`` 清理动作
   可以据此播报不同提示。
3. **需要注册清理回调**：例如 shell 动作在取消时要杀掉整个进程组。

令牌内部使用 ``threading.Event``，因此 **线程安全**，可以在任意线程检查和触发。
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class CancelToken:
    """可在线程间共享的取消令牌。

    线程安全：所有方法都可以在任意线程调用。回调在触发 :meth:`cancel` 的线程中同步执行，
    因此回调必须快速返回、不能阻塞。
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[str], None]] = []

    @property
    def cancelled(self) -> bool:
        """是否已被取消。"""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """取消原因（未取消时为空字符串）。"""
        return self._reason

    def cancel(self, reason: str = "") -> bool:
        """触发取消，并依次执行已注册的回调。

        Args:
            reason: 取消原因，例如 ``timeout`` / ``user`` / ``barge_in``。

        Returns:
            本次调用是否真正触发了取消（重复取消返回 ``False``）。
        """
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._event.set()
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(reason)
            except Exception:
                logger.exception("取消回调执行出错")
        return True

    def raise_if_cancelled(self) -> None:
        """如果已取消，抛出 :class:`asyncio.CancelledError`。

        在长循环中调用，让同步代码也能及时响应取消。
        """
        if self._event.is_set():
            raise asyncio.CancelledError(self._reason)

    def on_cancel(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """注册取消回调；如果已经取消，立即执行。

        Args:
            callback: 接收取消原因的回调函数。

        Returns:
            注销回调的函数。
        """
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)

                def remove() -> None:
                    with self._lock:
                        if callback in self._callbacks:
                            self._callbacks.remove(callback)

                return remove
        callback(self._reason)
        return lambda: None

    def wait(self, timeout: float | None = None) -> bool:
        """在线程中阻塞等待取消（可替代 ``time.sleep``，取消时立即醒来）。

        Returns:
            等待结束时是否已取消。
        """
        return self._event.wait(timeout)

    async def wait_async(self) -> str:
        """在协程中等待取消发生，返回取消原因。

        实现上是把阻塞等待放到线程里，避免轮询；主要用于测试和少量需要“等取消”的场景。
        """
        await asyncio.to_thread(self._event.wait)
        return self._reason


# 当前协程上下文中的取消令牌。TaskManager 启动任务时会设置它，
# 被托管的代码在任意深度都可以通过 current_token() 拿到，不必层层传参。
_current_token: contextvars.ContextVar[CancelToken | None] = contextvars.ContextVar(
    "pinecone_cancel_token", default=None
)


def current_token() -> CancelToken:
    """获取当前任务的取消令牌；不在托管任务中时返回一个永远不会被取消的新令牌。"""
    return _current_token.get() or CancelToken()


def set_current_token(token: CancelToken) -> contextvars.Token[CancelToken | None]:
    """设置当前上下文的取消令牌（由 TaskManager 内部调用）。"""
    return _current_token.set(token)
