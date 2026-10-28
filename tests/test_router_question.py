"""有未回答的提问时的路由：回答交给 AI，高优先级规则仍然先处理并作废提问。"""

from __future__ import annotations

from pinecone.agent.question import Choice, make_question
from pinecone.dialog import Reply
from pinecone.nlu.router import IntentRouter
from tests.conftest import SkillEnv


class _AskingAgent:
    """正在等用户回答的假 AI。"""

    def __init__(self) -> None:
        self.pending = make_question(
            [Choice(id="0", label="宝船，DOUDOU"), Choice(id="1", label="宝船，燕云十六声")],
            include_head=False,
        )
        self.texts: list[str] = []
        self.abandoned = 0

    async def pending_question(self):
        return self.pending

    async def abandon_question(self) -> None:
        self.abandoned += 1
        self.pending = None

    async def handle(self, text: str, *, hint: str | None = None) -> Reply:
        self.texts.append(text)
        return Reply(text="AI")


async def _router(env: SkillEnv) -> tuple[IntentRouter, _AskingAgent]:
    env.config.skills.enabled = ["task_control", "clock"]
    await env.manager.load()
    agent = _AskingAgent()
    return IntentRouter(env.manager, agent), agent


async def test_answers_and_free_replies_go_to_the_question(skill_env: SkillEnv) -> None:
    """序号、取消（含同时是停止指令的“算了”）和普通规则命中的话都交给提问。"""
    router, agent = await _router(skill_env)
    for text in ("第二个", "算了", "现在几点"):
        assert (await router.handle(text)).text == "AI"
    assert agent.texts == ["第二个", "算了", "现在几点"]
    assert agent.abandoned == 0


async def test_high_priority_rule_abandons_the_question(skill_env: SkillEnv) -> None:
    """“别说了”不是在回答：提问作废，由 task_control 处理，不交给 AI。"""
    router, agent = await _router(skill_env)
    reply = await router.handle("别说了")
    assert reply.text is None and skill_env.dialog.interrupted == 1
    assert agent.abandoned == 1 and agent.texts == []
