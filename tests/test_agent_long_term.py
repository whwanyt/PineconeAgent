"""长期记忆测试：增删查、注入提示词、工具调用流程。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

from pinecone.agent import VoiceAgent
from pinecone.agent.memory.long_term import LongTermMemory, relevance
from pinecone.core.config import AppConfig
from tests.agent_fakes import ScriptedChatModel


def test_relevance() -> None:
    assert relevance("我喜欢听什么歌", "用户喜欢听周杰伦的歌") > 0.3
    assert relevance("明天天气", "用户叫小明") == 0


async def test_add_search_forget(tmp_path: Path) -> None:
    async with AsyncSqliteStore.from_conn_string(str(tmp_path / "m.db")) as store:
        await store.setup()
        memory = LongTermMemory(store)
        assert await memory.add("用户叫小明")
        assert not await memory.add("用户叫小明")  # 不重复记录
        await memory.add("用户喜欢听周杰伦的歌")
        assert [m.text for m in await memory.search("喜欢听什么歌")] == ["用户喜欢听周杰伦的歌"]
        assert await memory.forget("用户叫小明") == ["用户叫小明"]
        assert [m.text for m in await memory.all()] == ["用户喜欢听周杰伦的歌"]


async def test_for_prompt_limits() -> None:
    memory = LongTermMemory(InMemoryStore(), inject_relevant=1, inject_recent=1)
    for text in ["用户叫小明", "用户喜欢周杰伦", "用户七点起床"]:
        await memory.add(text)
    picked = await memory.for_prompt("我叫什么名字小明")
    assert picked[0] == "用户叫小明" and len(picked) == 2


async def test_agent_remembers_via_tool_and_injects() -> None:
    """AI 调用 remember 工具；下一轮系统提示词中带上这条记忆。"""
    memory = LongTermMemory(InMemoryStore())
    call = AIMessage(
        content="", tool_calls=[{"name": "remember", "args": {"content": "用户叫小明"}, "id": "c1"}]
    )
    model = ScriptedChatModel.from_script(call, "好的，我记住了。", "你叫小明。")
    agent = VoiceAgent(AppConfig(), model, memory=memory)

    reply = await agent.handle("记住我叫小明")
    assert "".join([p async for p in reply.stream]) == "好的，我记住了。"
    assert [m.text for m in await memory.all()] == ["用户叫小明"]

    agent.session.reset()  # 新会话：短期记忆清空，长期记忆仍在
    reply = await agent.handle("我叫什么")
    assert "".join([p async for p in reply.stream]) == "你叫小明。"
    system = model.received[-1][0]
    assert isinstance(system, SystemMessage) and "- 用户叫小明" in system.content
