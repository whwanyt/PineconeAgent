"""AI Agent 测试（假模型，不访问网络）。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from pinecone.agent import VoiceAgent
from pinecone.agent.agent import Session
from pinecone.agent.prompts import build_system_prompt
from pinecone.core.config import AppConfig
from tests.agent_fakes import ScriptedChatModel


async def _collect(agent: VoiceAgent, text: str) -> str:
    reply = await agent.handle(text)
    assert reply.stream is not None and reply.follow_up
    return "".join([piece async for piece in reply.stream])


def test_system_prompt() -> None:
    prompt = build_system_prompt("小派", "说话温柔", ["用户叫小明"])
    assert "「小派」" in prompt and "说话温柔" in prompt and "- 用户叫小明" in prompt
    assert "当前时间" in prompt


async def test_stream_and_multi_turn() -> None:
    """流式输出文本；同一会话内保留上下文；系统提示词每次注入。"""
    model = ScriptedChatModel.from_script("你好呀，我是小派。", "你刚才问的是你好。")
    agent = VoiceAgent(AppConfig(), model)
    assert await _collect(agent, "你好") == "你好呀，我是小派。"
    assert await _collect(agent, "我刚才说了什么") == "你刚才问的是你好。"

    second_call = model.received[1]
    assert isinstance(second_call[0], SystemMessage) and "小派" in second_call[0].content
    assert [m.content for m in second_call[1:]] == ["你好", "你好呀，我是小派。", "我刚才说了什么"]


async def test_error_yields_apology() -> None:
    agent = VoiceAgent(AppConfig(), ScriptedChatModel.from_script(ConnectionError("断网")))
    assert "连不上大模型" in await _collect(agent, "你好")


async def test_repair_interrupted_tool_call() -> None:
    """工具调用被打断后，下一轮开始前补上“已取消”的工具结果。"""
    model = ScriptedChatModel.from_script("好的。")
    agent = VoiceAgent(AppConfig(), model)
    config = {"configurable": {"thread_id": agent.session.thread_id()}}
    dangling = AIMessage(content="", tool_calls=[{"name": "set_alarm", "args": {}, "id": "call_1"}])
    await agent._graph.aupdate_state(config, {"messages": [dangling]}, as_node="model")

    assert await _collect(agent, "算了") == "好的。"
    sent = model.received[0]
    tool_results = [m for m in sent if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_results] == ["call_1"]


async def test_cancel_mid_stream() -> None:
    """消费方被取消时，流随之结束，不会继续产出。"""
    agent = VoiceAgent(AppConfig(), ScriptedChatModel.from_script("一 二 三 四 五 六 七 八"))
    reply = await agent.handle("数数")
    assert reply.stream is not None
    pieces: list[str] = []

    async def consume() -> None:
        async for piece in reply.stream:
            pieces.append(piece)
            await asyncio.sleep(0.05)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.08)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert 0 < len(pieces) < 15


def test_session_rotation() -> None:
    session = Session(idle_min=0.0001)
    first = session.thread_id()
    session._last_active -= 1
    assert session.thread_id() != first or first.startswith("voice-")
    session.reset()
    assert session.thread_id().startswith("voice-")
