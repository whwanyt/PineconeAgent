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
不直接对用户说“没找到”，而是把原话交给兜底 AI，由它决定要不要用其他工具。没有配置 AI 时，
仍播报技能准备好的 reply。
"""

from __future__ import annotations

import logging

from pinecone.dialog import DialogHandler, Reply
from pinecone.nlu.matcher import normalize
from pinecone.skills.registry import SkillManager

logger = logging.getLogger(__name__)

_DEFAULT_UNKNOWN = "抱歉，我还不会这个。"


class IntentRouter:
    """实现 :class:`~pinecone.dialog.DialogHandler` 协议的路由器。"""

    def __init__(self, skills: SkillManager, fallback: DialogHandler | None = None) -> None:
        self._skills = skills
        self._fallback = fallback

    def set_fallback(self, fallback: DialogHandler | None) -> None:
        """设置兜底处理器（AI Agent）。"""
        self._fallback = fallback

    async def handle(self, text: str) -> Reply:
        """路由一句用户文本。运行在可取消的「对话」任务中。"""
        hit = self._skills.match(normalize(text))
        if hit is not None:
            skill, intent = hit
            intent.text = text
            logger.info("规则命中：%s.%s %s", skill.name, intent.name, intent.slots or "")
            result = await self._skills.execute(skill, intent)
            # 技能认领了这句话但处理不了（如本地没有这首歌）。有 AI 时把原话交回去，
            # 不播报“没找到”，否则用户听不到后续的查找或下载。
            if result.delegate and self._fallback is not None:
                logger.info("技能未处理，交给 AI：%s", text)
                return await self._fallback.handle(text)
            return Reply(text=result.reply, stream=result.stream, follow_up=result.follow_up)

        if self._fallback is not None:
            logger.info("规则未命中，交给 AI：%s", text)
            return await self._fallback.handle(text)
        return Reply(text=_DEFAULT_UNKNOWN, follow_up=False)
