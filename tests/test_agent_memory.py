"""Agent 短期记忆测试：SQLite 持久化、摘要压缩、过期清理。"""

from __future__ import annotations

import time
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from pinecone.agent import VoiceAgent
from pinecone.agent.memory.short_term import create_summarization, prune_history
from pinecone.core.config import AppConfig
from tests.agent_fakes import ScriptedChatModel


async def _ask(agent: VoiceAgent, text: str) -> str:
    reply = await agent.handle(text)
    assert reply.stream is not None
    return "".join([p async for p in reply.stream])


async def test_history_persists_across_restart(tmp_path: Path) -> None:
    db = str(tmp_path / "history.db")
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        agent = VoiceAgent(
            AppConfig(), ScriptedChatModel.from_script("记住了。"), checkpointer=saver
        )
        await _ask(agent, "我叫小明")
        thread_id = agent.session.thread_id()

    model = ScriptedChatModel.from_script("你叫小明。")
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        agent = VoiceAgent(AppConfig(), model, checkpointer=saver)
        agent.session._thread_id = thread_id  # 模拟仍在会话空闲期内
        assert await _ask(agent, "我叫什么") == "你叫小明。"
    assert [m.content for m in model.received[0][1:]] == ["我叫小明", "记住了。", "我叫什么"]


async def test_summarization_compresses_history() -> None:
    """历史超过阈值时较早的消息被压缩成摘要，只保留最近几条原文。"""
    model = ScriptedChatModel.from_script(
        "第一轮的回答，内容比较长一些。",
        "第二轮的回答，内容也比较长。",
        "用户聊了两轮天气。",
        "第三轮回答。",
    )
    summarization = create_summarization(model, summarize_after_tokens=30, keep_messages=2)
    agent = VoiceAgent(AppConfig(), model, middleware=[summarization])
    await _ask(agent, "第一个问题是关于天气的")
    await _ask(agent, "第二个问题还是关于天气的")
    assert await _ask(agent, "第三个问题") == "第三轮回答。"

    last_call = model.received[-1]
    contents = " ".join(str(m.content) for m in last_call)
    assert "用户聊了两轮天气。" in contents
    assert "第一个问题是关于天气的" not in contents
    assert "第三个问题" in contents


async def test_prune_history(tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    old_id = f"voice-{int(time.time()) - 10 * 86400}"
    new_id = f"voice-{int(time.time())}"
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        agent = VoiceAgent(
            AppConfig(), ScriptedChatModel.from_script("一", "二"), checkpointer=saver
        )
        for thread_id, text in ((old_id, "旧"), (new_id, "新")):
            agent.session._thread_id = thread_id
            agent.session._last_active = time.time()
            await _ask(agent, text)

    assert prune_history(db, keep_days=7) == 1
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        threads = {c.config["configurable"]["thread_id"] async for c in saver.alist(None)}
    assert threads == {new_id}
