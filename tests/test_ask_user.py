"""模型调用 ask_user 才暂停；工具自己返回多条结果时不暂停。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from pinecone.agent import VoiceAgent
from pinecone.agent.question import omit_repeated_lead
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
        content="搜到好几首，你要哪一个？",
        tool_calls=[
            {
                "name": "ask_user",
                "args": {"question": "搜到好几个版本，你想听哪一首？", "options": _OPTIONS},
                "id": "c1",
            }
        ],
    )


def test_omit_repeated_lead() -> None:
    """模型已经说过的开场白不再接在选项前面。"""
    prompt = (
        "搜到好几个版本，你想听哪首？。第1个，黄诗扶，九万字。第2个，张靓颖，九万字。你要哪一个？"
    )
    already = "本地曲库没有这首歌，我去网上找一下。搜到好几个版本，你想听哪首？"
    assert omit_repeated_lead(already, prompt) == (
        "第1个，黄诗扶，九万字。第2个，张靓颖，九万字。你要哪一个？"
    )
    assert omit_repeated_lead("", prompt) == prompt


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
    assert "搜到好几个版本，你想听哪一首" not in asked
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


async def test_free_reply_is_returned_to_the_model() -> None:
    """没说第几个时，不重问，把原话交给模型。"""
    model = ScriptedChatModel.from_script(_ask_call(), "好，给你放张靓颖的现场版。")
    agent = VoiceAgent(AppConfig(), model)
    await _spoken(agent, "播放宝船")
    reply = await _spoken(agent, "就听张靓颖的现场版")
    assert reply == "好，给你放张靓颖的现场版。"
    assert await agent.pending_question() is None
    tool_msg = next(message for message in model.received[-1] if isinstance(message, ToolMessage))
    assert "用户没有选第几个" in tool_msg.content
    assert "就听张靓颖的现场版" in tool_msg.content


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


async def test_abandoned_question_lets_the_next_turn_start_fresh() -> None:
    """提问作废后不再暂停，下一句作为新的用户消息处理，会话上下文保留。"""
    model = ScriptedChatModel.from_script(_ask_call(), "现在三点。")
    agent = VoiceAgent(AppConfig(), model)
    await _spoken(agent, "播放宝船")
    thread = agent.session.current()

    await agent.abandon_question()
    assert await agent.pending_question() is None
    assert await _spoken(agent, "几点了") == "现在三点。"
    assert agent.session.current() == thread
    tool_msg = next(message for message in model.received[-1] if isinstance(message, ToolMessage))
    assert "提问已作废" in tool_msg.content
    assert model.received[-1][-1].content == "几点了"


async def test_question_expires_with_the_session() -> None:
    """会话空闲超时后，旧提问不再接走下一句。"""
    model = ScriptedChatModel.from_script(_ask_call(), "现在三点。")
    agent = VoiceAgent(AppConfig(), model)
    await _spoken(agent, "播放宝船")
    agent.session._last_active -= agent.session._idle_s + 1

    assert await agent.pending_question() is None
    assert await _spoken(agent, "几点了") == "现在三点。"
    assert not any(isinstance(message, ToolMessage) for message in model.received[-1])


async def test_question_reply_expires_when_follow_up_times_out() -> None:
    """提问的回复带上 on_expire，追问窗口超时后由对话层调用。"""
    model = ScriptedChatModel.from_script(_ask_call())
    agent = VoiceAgent(AppConfig(), model)
    reply = await agent.handle("播放宝船")
    assert reply.stream is not None
    _ = [piece async for piece in reply.stream]
    assert reply.on_expire is not None

    await reply.on_expire()
    assert await agent.pending_question() is None
