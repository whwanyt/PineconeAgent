"""TaskManager / CancelToken 的单元测试。"""

import asyncio
import threading

from pinecone.core.bus import EventBus
from pinecone.core.cancel import CancelToken, current_token
from pinecone.core.events import TaskFinished
from pinecone.core.task_manager import TaskKind, TaskManager, TaskStatus


def make_manager() -> tuple[TaskManager, list[TaskFinished]]:
    bus = EventBus()
    finished: list[TaskFinished] = []
    bus.subscribe(TaskFinished, finished.append)
    return TaskManager(bus), finished


async def test_run_success_and_event() -> None:
    """正常完成的任务状态为 done，结果可读取，并发布 TaskFinished。"""
    tm, finished = make_manager()

    async def work() -> int:
        return 42

    rt = await tm.run(work(), name="计算")
    assert rt.status == TaskStatus.DONE and rt.result == 42
    assert finished[0].status == "done"
    assert tm.list() == []


async def test_cancel_runs_on_cancel_with_reason() -> None:
    """取消任务后执行 on_cancel，并能拿到取消原因；令牌中的回调也会触发。"""
    tm, _ = make_manager()
    reasons: list[str] = []
    token_reasons: list[str] = []

    async def work(token: CancelToken) -> None:
        token.on_cancel(token_reasons.append)
        await asyncio.sleep(10)

    async def cleanup(reason: str) -> None:
        reasons.append(reason)

    rt = tm.spawn(work, name="备份照片", on_cancel=cleanup)
    await asyncio.sleep(0.01)
    assert tm.cancel_by_name("备份") == [rt]
    assert await rt.wait() == TaskStatus.CANCELLED
    assert reasons == ["user"] and token_reasons == ["user"]


async def test_timeout() -> None:
    """超时的任务状态为 timeout，on_cancel 收到原因 timeout。"""
    tm, _ = make_manager()
    reasons: list[str] = []

    async def cleanup(reason: str) -> None:
        reasons.append(reason)

    rt = await tm.run(asyncio.sleep(10), name="慢任务", timeout=0.05, on_cancel=cleanup)
    assert rt.status == TaskStatus.TIMEOUT
    assert reasons == ["timeout"]


async def test_cancel_foreground_keeps_background() -> None:
    """打断只取消前台任务，后台任务和不可取消任务继续运行。"""
    tm, _ = make_manager()
    fg = tm.spawn(asyncio.sleep(10), name="播报")
    bg = tm.spawn(asyncio.sleep(10), name="音乐", kind=TaskKind.BACKGROUND)
    locked = tm.spawn(asyncio.sleep(10), name="系统", cancellable=False)
    await asyncio.sleep(0)

    assert tm.cancel_foreground() == 1
    await fg.wait()
    assert fg.status == TaskStatus.CANCELLED
    assert not bg.done and not locked.done

    await tm.shutdown()
    assert bg.status == TaskStatus.CANCELLED and locked.status == TaskStatus.CANCELLED


async def test_cancel_before_start() -> None:
    """任务刚 spawn 还没开始运行就被取消，也能正确清理。"""
    tm, finished = make_manager()
    rt = tm.spawn(asyncio.sleep(10), name="秒取消")
    rt.cancel()
    assert await rt.wait() == TaskStatus.CANCELLED
    await asyncio.sleep(0.01)
    assert tm.list() == [] and finished[0].status == "cancelled"


async def test_thread_code_sees_token() -> None:
    """线程中的同步代码通过 current_token 感知取消并提前退出。"""
    tm, _ = make_manager()
    exited = threading.Event()

    def blocking_loop(token: CancelToken) -> None:
        while not token.wait(0.01):
            pass
        exited.set()

    async def work() -> None:
        await asyncio.to_thread(blocking_loop, current_token())

    rt = tm.spawn(work(), name="线程任务")
    await asyncio.sleep(0.05)
    rt.cancel()
    await rt.wait()
    assert await asyncio.to_thread(exited.wait, 1)


async def test_current_follows_the_token_into_foreign_tasks() -> None:
    """库自己 create_task 出来的协程（如 LangGraph 跑工具）里，current() 仍能找到所在任务。"""
    tasks = TaskManager(EventBus())
    seen: list[object] = []

    async def tool() -> None:
        seen.append(tasks.current())
        child = tasks.spawn(asyncio.sleep(0), name="技能")
        seen.append(child.parent_id)
        await child.wait()

    async def turn() -> None:
        await asyncio.create_task(tool())

    rt = tasks.spawn(turn(), name="对话")
    await rt.wait()
    assert seen == [rt, rt.id]
