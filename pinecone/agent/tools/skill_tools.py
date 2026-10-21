"""把技能的 :class:`~pinecone.skills.ToolSpec` 转换成 LangChain ``StructuredTool``。

AI 调用工具时走的是和规则匹配 **完全相同** 的执行路径::

    StructuredTool(name=spec.name) ─▶ Intent(name=spec.intent, slots=固定槽位+参数, source="agent")
                                   ─▶ SkillManager.execute（托管任务、超时、取消、异常处理）
                                   ─▶ Skill.handle

因此技能只需实现一份业务逻辑；task_control 暴露的 ``list_tasks`` / ``cancel_task`` /
``stop_speaking`` 也会自动成为 AI 工具，可以说“把刚才那个下载停掉”。

工具返回给模型的是 JSON：``{"reply": 建议回复, "data": 结构化结果}``。
``reply`` 为空表示技能已经自己处理完（例如自定义任务已经播报过结果），模型简单确认即可。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from pinecone.skills import Intent, Skill, SkillManager, ToolSpec

logger = logging.getLogger(__name__)


class _NoArgs(BaseModel):
    """无参数工具的参数模型。"""


def _to_tool(skills: SkillManager, skill: Skill, spec: ToolSpec) -> BaseTool:
    async def run(**kwargs: Any) -> str:
        intent = Intent(
            skill=skill.name,
            name=spec.intent,
            slots={**spec.fixed_slots, **kwargs},
            source="agent",
        )
        logger.info("AI 调用工具 %s %s", spec.name, kwargs or "")
        result = await skills.execute(skill, intent)
        reply = result.reply
        if reply is None and result.stream is not None:
            reply = "".join([piece async for piece in result.stream])
        return json.dumps(
            {"reply": reply or "", "data": result.data}, ensure_ascii=False, default=str
        )

    return StructuredTool.from_function(
        coroutine=run,
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_schema or _NoArgs,
    )


def skill_tools(skills: SkillManager) -> list[BaseTool]:
    """收集全部已加载技能暴露的工具（重名时保留优先级高的技能的工具）。"""
    tools: dict[str, BaseTool] = {}
    for skill in skills.skills:  # 已按优先级从高到低排序
        for spec in skill.tools():
            if spec.name in tools:
                logger.warning("工具名重复，忽略 %s.%s", skill.name, spec.name)
                continue
            tools[spec.name] = _to_tool(skills, skill, spec)
    return list(tools.values())
