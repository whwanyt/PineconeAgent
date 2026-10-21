"""事件总线与状态机的单元测试。"""

import asyncio

import pytest

from pinecone.core.bus import EventBus
from pinecone.core.events import Event, StateChanged, Transcribed
from pinecone.core.state import AssistantState, InvalidTransition, StateMachine


async def test_publish_to_sync_async_and_parent_handlers() -> None:
    """同步、异步处理函数都能收到事件；订阅父类 Event 可以收到所有事件。"""
    bus = EventBus()
    got: list[str] = []

    async def on_async(e: Transcribed) -> None:
        got.append(f"async:{e.text}")

    bus.subscribe(Transcribed, lambda e: got.append(f"sync:{e.text}"))
    bus.subscribe(Transcribed, on_async)
    bus.subscribe(Event, lambda e: got.append("any"))

    await bus.publish(Transcribed(text="你好"))
    assert got == ["sync:你好", "async:你好", "any"]


async def test_handler_error_does_not_break_others() -> None:
    """一个处理函数抛异常，不影响后续处理函数。"""
    bus = EventBus()
    got: list[int] = []

    def bad(_: Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe(Transcribed, bad)
    bus.subscribe(Transcribed, lambda e: got.append(1))
    await bus.publish(Transcribed(text="x"))
    assert got == [1]


async def test_wait_for_and_threadsafe_publish() -> None:
    """从其他线程投递的事件能被 wait_for 等到。"""
    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())

    waiter = asyncio.create_task(bus.wait_for(Transcribed, lambda e: e.text == "目标"))
    await asyncio.sleep(0)
    await asyncio.to_thread(bus.publish_threadsafe, Transcribed(text="其他"))
    await asyncio.to_thread(bus.publish_threadsafe, Transcribed(text="目标"))
    event = await asyncio.wait_for(waiter, 1)
    assert event.text == "目标"


async def test_state_machine_transitions() -> None:
    """合法迁移发布 StateChanged，非法迁移抛异常，任意状态可回到 IDLE。"""
    bus = EventBus()
    sm = StateMachine(bus)
    changes: list[tuple[str, str]] = []
    bus.subscribe(StateChanged, lambda e: changes.append((e.old, e.new)))

    await sm.transition(AssistantState.LISTENING)
    with pytest.raises(InvalidTransition):
        # LISTENING 只能进入 THINKING（或回到 IDLE）
        await sm.transition(AssistantState.SPEAKING)
    await sm.transition(AssistantState.THINKING)
    assert sm.is_busy()
    await sm.transition(AssistantState.IDLE)
    assert changes == [("idle", "listening"), ("listening", "thinking"), ("thinking", "idle")]
