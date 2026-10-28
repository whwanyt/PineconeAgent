"""意图路由：规则匹配优先，未命中时交给兜底处理器（AI Agent）。

路由优先级（由技能的 ``priority`` 决定）::

    自定义任务关键词（custom_task，100）
        > 任务控制（task_control，90：“停止”“取消XX”）
        > 普通技能正则（50）
        > AI Agent（兜底，可通过工具调用任意技能）

为什么不全部交给大模型？
    规则匹配在本地瞬间完成、零费用、断网可用，“几点了”“停止”这类高频指令体验最好；
    大模型负责规则覆盖不到的开放式问题和复杂表达。

技能可以在规则命中后仍表示自己处理不了（``SkillResult.delegate``）。例如本地曲库没有这首歌时，
不直接对用户说“没找到”，而是把原话和 ``hint``（情况说明）交给兜底 AI，由模型先开口、
再决定要不要用其他工具。没有配置 AI 时，仍播报技能准备好的 reply。

AI 正在等用户回答提问（``ask_user``）时，这句原话默认交给 AI，连“就听现场版”这种
会被音乐规则抢走的话也一样。例外是没在回答、却命中了高优先级规则（“停止”、自定义任务）的话：
提问作废，按规则处理，本地指令不必联网、也不会被模型误解成选项。
"""

from __future__ import annotations

import logging

from pinecone.dialog import DialogHandler, Reply
from pinecone.nlu.matcher import normalize
from pinecone.skills.base import Intent, Skill
from pinecone.skills.registry import SkillManager

logger = logging.getLogger(__name__)

_DEFAULT_UNKNOWN = "抱歉，我还不会这个。"
_PREEMPT_PRIORITY = 90
"""有未回答的提问时，优先级不低于这个数的规则仍然先处理（task_control、custom_task）。"""


class IntentRouter:
    """实现 :class:`~pinecone.dialog.DialogHandler` 协议的路由器。"""

    def __init__(self, skills: SkillManager, fallback: DialogHandler | None = None) -> None:
        self._skills = skills
        self._fallback = fallback

    def set_fallback(self, fallback: DialogHandler | None) -> None:
        """设置兜底处理器（AI Agent）。"""
        self._fallback = fallback

    async def handle(self, text: str, *, hint: str | None = None) -> Reply:
        """路由一句用户文本。运行在可取消的「对话」任务中。

        Args:
            text: 用户原话。
            hint: 上层传入的情况说明；规则未命中时原样交给兜底 AI。
        """
        hit = self._skills.match(normalize(text))
        if self._fallback is not None and await self._answers_question(text, hit):
            logger.info("回答提问：%s", text)
            return await self._fallback.handle(text)

        if hit is not None:
            skill, intent = hit
            intent.text = text
            logger.info("规则命中：%s.%s %s", skill.name, intent.name, intent.slots or "")
            result = await self._skills.execute(skill, intent)
            # 技能认领了这句话但处理不了（如本地没有这首歌）。有 AI 时把原话和
            # hint 交回去，不播报“没找到”，由模型自己告诉用户正在做什么。
            if result.delegate and self._fallback is not None:
                logger.info("技能未处理，交给 AI：%s", text)
                return await self._fallback.handle(text, hint=result.hint)
            # 没有播报时留一句执行说明，对话层把它和用户原话写进同一条会话历史。
            # 有播报时以实际说出的话为准，说明留空。
            note = ""
            if not result.reply and result.stream is None:
                note = _execution_note(skill, intent)
            return Reply(
                text=result.reply,
                stream=result.stream,
                follow_up=result.follow_up,
                note=note,
            )

        if self._fallback is not None:
            logger.info("规则未命中，交给 AI：%s", text)
            return await self._fallback.handle(text, hint=hint)
        return Reply(text=_DEFAULT_UNKNOWN, follow_up=False)

    async def _answers_question(self, text: str, hit: tuple[Skill, Intent] | None) -> bool:
        """有未回答的提问时，判断这句是否交给 AI 当作回答。

        高优先级规则命中、又不像在回答时返回 False，并作废这次提问。
        """
        agent = self._fallback
        pending_of = getattr(agent, "pending_question", None)
        if pending_of is None:
            return False
        pending = await pending_of()
        if pending is None:
            return False
        if hit is None or hit[0].priority < _PREEMPT_PRIORITY or pending.is_answer(text):
            return True
        logger.info("没有回答提问，改由技能 %s 处理：%s", hit[0].name, text)
        await agent.abandon_question()
        return False


def _execution_note(skill: Skill, intent: Intent) -> str:
    """技能做完了但没有出声时，写进会话历史的那一句说明。"""
    details = [f"{key}={value}" for key, value in intent.slots.items() if value not in (None, "")]
    action = intent.name if not details else f"{intent.name}，{'，'.join(details)}"
    label = skill.description or skill.name
    return f"已执行{label}（{action}），没有额外播报。"
