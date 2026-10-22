"""模型调用 ask_user 才暂停；工具自己返回多条结果时不暂停。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from pinecone.agent import VoiceAgent
from pinecone.core.config import AppConfig
from tests.agent_fakes import ScriptedChatModel

_OPTIONS = ["宝船，DOUDOU", "宝船，燕云十六声"]


class _NoArgs(BaseModel):
    """无参数工具。"""


def _list_songs_tool() -> StructuredTool:
    async def list_songs() -> str:
        return json.dumps({"songs": _OPTIONS}, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=list_songs,
        name="list_songs",
        description="搜歌，可能返回多首",
        args_schema=_NoArgs,
    )


def _ask_call() -> AIMessage:
    return AIMessage(
        content="搜到好几首。",
        tool_calls=[
            {
                "name": "ask_user",
                "args": {"question": "搜到好几首", "options": _OPTIONS},
                "id": "c1",
            }
        ],
    )


async def _spoken(agent: VoiceAgent, text: str) -> str:
    reply = await agent.handle(text)
    if reply.stream is None:
        return reply.text or ""
    return "".join([piece async for piece in reply.stream])


async def test_ask_user_pauses_until_the_user_picks() -> None:
    """模型调用 ask_user 后只念选项；「第二个」把对应选项交回模型。"""
    model = ScriptedChatModel.from_script(_ask_call(), "好的，播放燕云十六声的宝船。")
    agent = VoiceAgent(AppConfig(), model)
    asked = await _spoken(agent, "播放宝船")
    assert "第1个，宝船，DOUDOU" in asked
    assert "第2个，宝船，燕云十六声" in asked
    assert "你要哪一个" in asked
    assert "好的，播放燕云十六声" not in asked
    assert await agent.pending_question() is not None

    chosen = await _spoken(agent, "第二个")
    assert chosen == "好的，播放燕云十六声的宝船。"
    assert await agent.pending_question() is None
    tool_msg = next(message for message in model.received[-1] if isinstance(message, ToolMessage))
    assert "用户选择了第2个：宝船，燕云十六声" in tool_msg.content


async def test_cancel_ask_user() -> None:
    """取消后模型收到取消说明，图不再暂停。"""
    model = ScriptedChatModel.from_script(_ask_call(), "好的，那就不放了。")
    agent = VoiceAgent(AppConfig(), model)
    await _spoken(agent, "播放宝船")
    reply = await _spoken(agent, "取消")
    assert reply == "好的，那就不放了。"
    tool_msg = next(message for message in model.received[-1] if isinstance(message, ToolMessage))
    assert "用户取消了选择" in tool_msg.content


async def test_many_results_do_not_pause_unless_model_asks() -> None:
    """工具返回多首、模型没有调用 ask_user 时，直接把模型的话念出来，图不暂停。"""
    call = AIMessage(
        content="",
        tool_calls=[{"name": "list_songs", "args": {}, "id": "c1"}],
    )
    model = ScriptedChatModel.from_script(call, "我放第一首，DOUDOU 的宝船。")
    agent = VoiceAgent(AppConfig(), model, tools=[_list_songs_tool()])
    spoken = await _spoken(agent, "播放宝船")
    assert spoken == "我放第一首，DOUDOU 的宝船。"
    assert await agent.pending_question() is None
    assert "ask_user" not in {
        message.name for message in model.received[-1] if isinstance(message, ToolMessage)
    }
