"""规则直接命中的技能回合，和 AI 对话写进同一条会话历史。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from pinecone.agent import VoiceAgent
from pinecone.core.events import Spoken
from pinecone.core.state import StateMachine
from pinecone.dialog import DialogController, Reply
from pinecone.nlu.router import IntentRouter
from pinecone.skills.builtin.music.player import MusicPlayer, PlayState
from tests.agent_fakes import ScriptedChatModel
from tests.conftest import SkillEnv


class _Player(MusicPlayer):
    """技能加载时要有一个播放器；本测试不真正放歌。"""

    def load(self, uri: str) -> None:
        return None

    def pause(self) -> None:
        return None

    def resume(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def set_volume(self, volume: int) -> None:
        return None

    @property
    def state(self) -> PlayState:
        return PlayState.IDLE

    @property
    def paused(self) -> bool:
        return False


class _Speaker:
    """播报时发布 Spoken，对话层据此把说出的话写入会话。"""

    def __init__(self, bus) -> None:
        self.bus = bus
        self.said: list[str] = []
        self._n = 0

    async def say(self, text: str) -> str:
        self._n += 1
        self.said.append(text)
        await self.bus.publish(Spoken(text=text, utterance=self._n))
        return text

    async def say_stream(self, pieces) -> str:
        return await self.say("".join([piece async for piece in pieces]))

    def stop(self) -> None:
        return None


class _Output:
    async def play(self, samples, rate) -> None:
        return None


def _dialog(env: SkillEnv, handler, agent: VoiceAgent) -> DialogController:
    env.config.dialog.wake_beep = False
    env.config.dialog.follow_up_s = 0
    env.bus.bind_loop(asyncio.get_running_loop())
    dialog = DialogController(
        config=env.config,
        bus=env.bus,
        state=StateMachine(env.bus),
        tasks=env.tasks,
        speaker=_Speaker(env.bus),  # type: ignore[arg-type]
        output=_Output(),  # type: ignore[arg-type]
        handler=handler,
    )
    dialog.set_exchange_log(agent)
    dialog.start()
    return dialog


async def _messages(agent: VoiceAgent) -> list:
    state = await agent._graph.aget_state({"configurable": {"thread_id": agent.session.current()}})
    return list(state.values.get("messages", []))


async def test_skill_turns_join_the_agent_thread(skill_env: SkillEnv) -> None:
    """看时间和没出声的停止都留在会话里，下一句交给模型时能看见。"""
    skill_env.config.skills.enabled = ["clock", "task_control"]
    await skill_env.manager.load()
    model = ScriptedChatModel.from_script("你刚才问了时间，然后让我停下。")
    agent = VoiceAgent(skill_env.config, model)
    dialog = _dialog(skill_env, IntentRouter(skill_env.manager, agent), agent)

    await dialog.handle_text("现在几点")
    await dialog.handle_text("别说了")
    await dialog.handle_text("我刚才说了什么")

    plain = [m.content for m in model.received[0] if not isinstance(m, SystemMessage)]
    assert plain[0] == "现在几点"
    assert str(plain[1]).startswith("现在是")
    assert plain[2] == "别说了"
    assert "没有额外播报" in str(plain[3])
    assert plain[-1] == "我刚才说了什么"
    await skill_env.tasks.shutdown()


async def test_spoken_during_the_skill_is_what_gets_stored(skill_env: SkillEnv) -> None:
    """技能自己说出口的话优先于执行说明。"""

    class _Saying:
        def __init__(self) -> None:
            self.bus = skill_env.bus

        async def handle(self, text: str, *, hint: str | None = None) -> Reply:
            await self.bus.publish(Spoken(text="灯已打开", utterance=1))
            return Reply(note="不该用这句")

    model = ScriptedChatModel.from_script("灯开过了。")
    agent = VoiceAgent(skill_env.config, model)
    dialog = _dialog(skill_env, _Saying(), agent)
    await dialog.handle_text("开灯")
    dialog.set_handler(agent)
    await dialog.handle_text("刚才怎样")
    plain = [m.content for m in model.received[0] if not isinstance(m, SystemMessage)]
    assert plain[:2] == ["开灯", "灯已打开"]
    await skill_env.tasks.shutdown()


async def test_delegate_is_written_only_by_the_agent(skill_env: SkillEnv, monkeypatch) -> None:
    """本地找不到歌、转交给 AI 时，用户原话只由模型写入一次。"""
    monkeypatch.setattr(
        "pinecone.skills.builtin.music.player.create_player",
        lambda device=None: _Player(),
    )
    skill_env.config.skills.enabled = ["music"]
    skill_env.config.skills.settings["music"] = {"library": [str(skill_env.config.data_path)]}
    await skill_env.manager.load()
    model = ScriptedChatModel.from_script("我去找找孙燕姿。")
    agent = VoiceAgent(skill_env.config, model)
    dialog = _dialog(skill_env, IntentRouter(skill_env.manager, agent), agent)

    await dialog.handle_text("我想听孙燕姿的歌")
    humans = [m.content for m in await _messages(agent) if isinstance(m, HumanMessage)]
    assert humans == ["我想听孙燕姿的歌"]
    await skill_env.tasks.shutdown()


async def test_pending_question_is_not_overwritten(skill_env: SkillEnv) -> None:
    """还停在提问上时不补写消息；回答之后历史里仍然只有原来那句用户消息。"""
    from tests.test_ask_user import _ask_call

    skill_env.config.skills.enabled = ["clock"]
    skill_env.config.agent.choice_timeout_s = 30
    await skill_env.manager.load()
    model = ScriptedChatModel.from_script(_ask_call(), "好的，放第一首。")
    agent = VoiceAgent(skill_env.config, model)
    dialog = _dialog(skill_env, IntentRouter(skill_env.manager, agent), agent)

    await dialog.handle_text("播放宝船")
    await agent.record_exchange("别插一句", "不该写进去")
    assert "别插一句" not in [m.content for m in await _messages(agent)]

    await dialog.handle_text("第一个")
    humans = [m.content for m in await _messages(agent) if isinstance(m, HumanMessage)]
    tools = [m for m in await _messages(agent) if isinstance(m, ToolMessage)]
    assert humans == ["播放宝船"]
    assert tools and "晴天" not in humans
    dialog._cancel_timer()
    await skill_env.tasks.shutdown()
