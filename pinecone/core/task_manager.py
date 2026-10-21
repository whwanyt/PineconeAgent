"""任务托管：所有“正在执行的事情”的统一登记处与终止入口。

为什么要统一托管？
    音箱需要支持“停止”“取消备份”“停止所有任务”“现在在做什么”等语音命令，
    以及唤醒词打断。如果各模块各自 ``asyncio.create_task``，就没有地方能列出并终止它们。
    因此约定：**技能调用、自定义任务、Agent 推理、TTS 播报、音乐、闹钟响铃**
    全部通过 :meth:`TaskManager.spawn` 启动。

任务类型：
    - ``FOREGROUND`` 前台：与当前对话相关（播报、Agent 推理、技能处理），
      唤醒词打断或说“停止”时会被取消；
    - ``BACKGROUND`` 后台：不阻塞对话（音乐、下载、备份等长任务），
      只有明确点名取消或“停止所有任务”时才会被取消。

取消传播链路::

    RunningTask.cancel(reason)
        ├─ token.cancel(reason)   → 触发 CancelToken 回调（如杀进程组），线程中的代码也能感知
        └─ asyncio_task.cancel()  → 协程中抛出 CancelledError，层层 finally 清理资源
    任务结束后：执行 on_cancel 清理回调（仅取消/超时时） → 发布 TaskFinished 事件
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pinecone.core.bus import EventBus
from pinecone.core.cancel import CancelToken, set_current_token
from pinecone.core.events import TaskFinished, TaskStarted

logger = logging.getLogger(__name__)

# on_cancel 清理回调的最长执行时间，防止清理动作本身卡住
ON_CANCEL_TIMEOUT = 10.0


class TaskKind(StrEnum):
    """任务类型。"""

    FOREGROUND = "foreground"
    BACKGROUND = "background"


class TaskStatus(StrEnum):
    """任务状态。"""

    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    FAILED = "failed"


# spawn 接受两种形式：
#   1. 直接传入协程对象：spawn(foo(), ...)
#   2. 传入“接收 CancelToken 返回协程”的函数：spawn(lambda token: foo(token), ...)
TaskTarget = Coroutine[Any, Any, Any] | Callable[[CancelToken], Awaitable[Any]]
OnCancel = Callable[[str], Awaitable[None]]


@dataclass(eq=False)
class RunningTask:
    """一个被托管的任务。

    对外只读的属性：``id`` / ``name`` / ``kind`` / ``source`` / ``status`` / ``elapsed`` 等；
    对外可调用的方法：:meth:`cancel`、:meth:`wait`。
    """

    id: str
    name: str
    kind: TaskKind
    source: str
    """任务来源，例如 ``skill:clock``、``task:备份``、``agent``、``tts``。"""
    timeout: float | None
    cancellable: bool
    token: CancelToken = field(default_factory=CancelToken)
    started_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.RUNNING
    error: str | None = None
    result: Any = None
    _task: asyncio.Task[Any] | None = field(default=None, repr=False)

    @property
    def elapsed(self) -> float:
        """已运行秒数。"""
        return time.time() - self.started_at

    @property
    def done(self) -> bool:
        """是否已结束（无论成功、失败还是取消）。"""
        return self.status != TaskStatus.RUNNING

    def cancel(self, reason: str = "user", *, force: bool = False) -> bool:
        """请求取消任务。

        Args:
            reason: 取消原因，会传给 on_cancel 回调。
            force: 是否强制取消声明为 ``cancellable=False`` 的任务（仅用于程序退出）。

        Returns:
            是否成功发出取消请求。已结束或不可取消的任务返回 ``False``。
        """
        if self.done or (not self.cancellable and not force):
            return False
        logger.info("取消任务 %s「%s」，原因：%s", self.id, self.name, reason)
        # 先触发令牌，让线程中的代码和取消回调（如杀进程）尽快生效，再取消协程
        self.token.cancel(reason)
        if self._task is not None:
            self._task.cancel(reason)
        return True

    async def wait(self) -> TaskStatus:
        """等待任务结束并返回最终状态。不会抛出任务内部的异常。"""
        if self._task is not None:
            await asyncio.wait({self._task})
        return self.status


class TaskManager:
    """任务托管器。

    非线程安全：只能在事件循环线程中调用（音频线程请通过事件总线间接触发）。
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._tasks: dict[str, RunningTask] = {}
        self._ids = itertools.count(1)

    # ------------------------------------------------------------------ 启动

    def spawn(
        self,
        target: TaskTarget,
        *,
        name: str,
        kind: TaskKind = TaskKind.FOREGROUND,
        source: str = "",
        timeout: float | None = None,
        cancellable: bool = True,
        on_cancel: OnCancel | None = None,
    ) -> RunningTask:
        """启动并托管一个任务。

        Args:
            target: 协程对象，或接收 :class:`CancelToken` 并返回可等待对象的函数。
            name: 人类可读的任务名（中文），用于“取消XX”的语音匹配和列表播报。
            kind: 前台 / 后台。
            source: 任务来源标识。
            timeout: 超时秒数，超时后自动取消，状态为 ``timeout``。
            cancellable: 是否允许被用户取消。
            on_cancel: 被取消或超时后执行的清理协程函数，参数为取消原因。

        Returns:
            :class:`RunningTask` 句柄。
        """
        rt = RunningTask(
            id=f"t{next(self._ids)}",
            name=name,
            kind=kind,
            source=source,
            timeout=timeout,
            cancellable=cancellable,
        )
        rt._task = asyncio.get_running_loop().create_task(
            self._run(rt, target, on_cancel), name=f"{rt.id}:{name}"
        )
        rt._task.add_done_callback(lambda _: self._on_task_done(rt, target))
        self._tasks[rt.id] = rt
        logger.debug("启动任务 %s「%s」(%s)", rt.id, name, kind)
        return rt

    def _on_task_done(self, rt: RunningTask, target: TaskTarget) -> None:
        """兜底处理：任务在包装器开始执行之前就被取消的情况。

        asyncio 中，一个还没开始运行的 Task 被 cancel 时，CancelledError 会在协程的第一行
        之前抛出，``_run`` 里的 try/except 根本没有机会执行。这里补上状态与登记表的清理。
        """
        if rt.status != TaskStatus.RUNNING:
            return
        rt.status = TaskStatus.CANCELLED
        self._tasks.pop(rt.id, None)
        if isinstance(target, Coroutine):
            target.close()
        asyncio.get_running_loop().create_task(
            self._bus.publish(
                TaskFinished(task_id=rt.id, name=rt.name, kind=rt.kind.value, status="cancelled")
            )
        )

    async def run(self, target: TaskTarget, **kwargs: Any) -> RunningTask:
        """启动任务并等待其结束，返回句柄（通过 ``status`` / ``result`` 查看结果）。

        参数同 :meth:`spawn`。如果调用方自身被取消，被托管的任务也会一并取消。
        """
        rt = self.spawn(target, **kwargs)
        try:
            await rt.wait()
        except asyncio.CancelledError:
            rt.cancel("caller_cancelled", force=True)
            raise
        return rt

    async def _run(self, rt: RunningTask, target: TaskTarget, on_cancel: OnCancel | None) -> None:
        """任务包装器：设置上下文令牌、处理超时 / 取消 / 异常、执行清理、发布事件。"""
        # contextvar 在当前 Task 的上下文副本中设置，不会影响其他任务
        set_current_token(rt.token)
        await self._bus.publish(TaskStarted(task_id=rt.id, name=rt.name, kind=rt.kind.value))

        timeout_cm = asyncio.timeout(rt.timeout)
        try:
            awaitable = target(rt.token) if callable(target) else target
            async with timeout_cm:
                rt.result = await awaitable
            rt.status = TaskStatus.DONE
        except TimeoutError:
            # 区分“我们设置的超时到期”和“任务内部自己抛出的 TimeoutError”
            if timeout_cm.expired():
                rt.status = TaskStatus.TIMEOUT
                rt.token.cancel("timeout")
                logger.warning("任务 %s「%s」超时（%.0fs）", rt.id, rt.name, rt.timeout)
            else:
                rt.status = TaskStatus.FAILED
                rt.error = "TimeoutError"
                logger.exception("任务 %s「%s」执行失败", rt.id, rt.name)
        except asyncio.CancelledError:
            rt.status = TaskStatus.CANCELLED
            # 不再向上抛：包装器本身就是任务的最外层，取消已被“消化”为 CANCELLED 状态。
            # 但要清掉当前 Task 的取消计数，否则后续的 await（on_cancel、publish）会立即再次被取消。
            current = asyncio.current_task()
            if current is not None:
                while current.uncancel() > 0:
                    pass
        except Exception as exc:
            rt.status = TaskStatus.FAILED
            rt.error = f"{type(exc).__name__}: {exc}"
            logger.exception("任务 %s「%s」执行失败", rt.id, rt.name)
        finally:
            if isinstance(target, Coroutine) and rt.status != TaskStatus.DONE:
                # 协程对象在被 await 之前就出错时需要手动关闭，避免 "never awaited" 警告
                target.close()

        if rt.status in (TaskStatus.CANCELLED, TaskStatus.TIMEOUT) and on_cancel is not None:
            reason = rt.token.reason or rt.status.value
            try:
                async with asyncio.timeout(ON_CANCEL_TIMEOUT):
                    await on_cancel(reason)
            except Exception:
                logger.exception("任务 %s「%s」的 on_cancel 清理失败", rt.id, rt.name)

        self._tasks.pop(rt.id, None)
        await self._bus.publish(
            TaskFinished(
                task_id=rt.id,
                name=rt.name,
                kind=rt.kind.value,
                status=rt.status.value,
                error=rt.error,
                result=rt.result,
            )
        )

    # ------------------------------------------------------------------ 查询

    def list(self, kind: TaskKind | None = None) -> list[RunningTask]:
        """列出正在运行的任务（按启动时间排序）。"""
        tasks = sorted(self._tasks.values(), key=lambda t: t.started_at)
        return [t for t in tasks if kind is None or t.kind == kind]

    def get(self, task_id: str) -> RunningTask | None:
        """按 id 获取正在运行的任务。"""
        return self._tasks.get(task_id)

    def find(self, query: str) -> list[RunningTask]:
        """按名称模糊查找正在运行的任务。

        匹配规则（依次放宽）：名称完全相同 > 名称包含查询词 > 查询词包含名称。
        例如“取消备份”中提取出的“备份”能匹配任务“备份照片”。
        """
        query = query.strip()
        if not query:
            return []
        tasks = self.list()
        for match in (
            lambda t: t.name == query,
            lambda t: query in t.name,
            lambda t: t.name in query,
        ):
            found = [t for t in tasks if match(t)]
            if found:
                return found
        return []

    # ------------------------------------------------------------------ 取消

    def cancel(self, task_id: str, reason: str = "user") -> bool:
        """按 id 取消任务。"""
        rt = self._tasks.get(task_id)
        return rt.cancel(reason) if rt else False

    def cancel_by_name(self, query: str, reason: str = "user") -> list[RunningTask]:
        """按名称模糊匹配并取消，返回被取消的任务列表。"""
        return [t for t in self.find(query) if t.cancel(reason)]

    def cancel_foreground(
        self, reason: str = "barge_in", *, exclude: set[str] | None = None
    ) -> int:
        """取消所有前台任务（打断 / “停止”时使用），后台任务不受影响。

        Args:
            reason: 取消原因。
            exclude: 不取消的任务 id（例如发起取消的任务本身）。

        Returns:
            被取消的任务数量。
        """
        exclude = exclude or set()
        return sum(t.cancel(reason) for t in self.list(TaskKind.FOREGROUND) if t.id not in exclude)

    def cancel_all(self, reason: str = "user", *, exclude: set[str] | None = None) -> int:
        """取消所有可取消任务，返回数量。"""
        exclude = exclude or set()
        return sum(t.cancel(reason) for t in self.list() if t.id not in exclude)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """程序退出时强制取消全部任务并等待它们结束。"""
        tasks = self.list()
        for t in tasks:
            t.cancel("shutdown", force=True)
        pending = [t._task for t in tasks if t._task is not None]
        if pending:
            await asyncio.wait(pending, timeout=timeout)
