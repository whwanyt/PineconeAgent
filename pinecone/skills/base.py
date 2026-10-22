"""技能基类与相关数据结构。

一个技能需要声明：
    - ``name`` / ``description``：唯一名称与功能描述（描述也会提供给 AI，帮助它决定何时调用）；
    - ``patterns``：规则匹配用的正则，键为意图名，值为正则列表。命中后正则的
      **命名分组** 会成为意图的 ``slots``（槽位），例如 ``(?P<time>.+)叫我``；
    - :meth:`handle`：处理意图并返回 :class:`SkillResult`；
    - 可选 :meth:`tools`：把技能能力以 :class:`ToolSpec` 形式暴露给 AI Agent。

规则匹配与 AI 工具调用走 **同一个** :meth:`handle` 入口：
    - 规则命中：``Intent(source="rule", slots=正则命名分组)``；
    - AI 调用工具：``Intent(source="agent", slots=工具参数)``。
因此技能只需要实现一份业务逻辑。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel

from pinecone.core.task_manager import TaskKind

if TYPE_CHECKING:
    from pinecone.skills.context import SkillContext


@dataclass
class Intent:
    """一次技能调用的意图。"""

    skill: str
    """目标技能名。"""
    name: str
    """意图名（技能内部区分不同功能，例如 ``set`` / ``cancel`` / ``list``）。"""
    text: str = ""
    """用户原话（AI 调用工具时可能为空）。正则匹配使用的是归一化文本，槽位取自归一化文本。"""
    slots: dict[str, Any] = field(default_factory=dict)
    """槽位：正则命名分组或 AI 工具参数。"""
    source: Literal["rule", "agent", "task"] = "rule"
    """意图来源：规则匹配 / AI 工具调用 / 自定义任务。"""

    def slot(self, key: str, default: Any = None) -> Any:
        """读取槽位，空字符串视为不存在。"""
        value = self.slots.get(key)
        return default if value in (None, "") else value


@dataclass
class SkillResult:
    """技能执行结果。"""

    reply: str | None = None
    """要播报的回复文本；为空表示不播报。"""
    stream: AsyncIterable[str] | None = None
    """流式回复（少数需要边生成边播报的技能使用）。"""
    follow_up: bool = False
    """播报完是否进入追问窗口。查询类技能一般不需要。"""
    data: dict[str, Any] = field(default_factory=dict)
    """结构化结果，AI 调用工具时会和 reply 一起返回给模型。"""
    delegate: bool = False
    """规则已经命中，但技能没有真正处理。路由器在配置了兜底 AI 时不播报 reply，
    把用户原话交给 AI。没有兜底时仍播报 reply。AI 工具调用不看这个字段，避免来回转交。"""
    hint: str | None = None
    """转交给兜底 AI 时的情况说明（例如本地曲库没找到）。框架不播报这句，
    只注入当轮模型上下文，由模型自己开口告诉用户正在做什么。"""


@dataclass
class ToolSpec:
    """技能暴露给 AI Agent 的一个工具。

    AI 调用该工具时，框架会构造 ``Intent(name=intent, slots=参数, source="agent")``
    并调用技能的 :meth:`Skill.handle`。
    """

    name: str
    """工具名（英文、下划线，全局唯一），例如 ``set_alarm``。"""
    description: str
    """工具说明（中文），告诉 AI 何时使用、参数含义。"""
    intent: str
    """对应的技能意图名。"""
    args_schema: type[BaseModel] | None = None
    """参数模型（pydantic），字段的 description 会提供给 AI。无参数时为 None。"""
    fixed_slots: dict[str, Any] = field(default_factory=dict)
    """固定槽位，与 AI 参数合并后传给 handle。用于一个意图对应多个工具的场景，
    例如每个自定义任务一个工具，都走 ``run`` 意图，靠 ``{"task": 任务名}`` 区分。"""


class Skill(ABC):
    """技能基类。

    子类必须是可无参实例化的（框架负责创建），私有配置通过 ``ctx.settings`` 读取。
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    patterns: ClassVar[dict[str, list[str]]] = {}
    """意图名 → 正则列表。正则在归一化文本上做 ``re.search``（不含标点、空格）。"""
    priority: ClassVar[int] = 50
    """匹配优先级，数字越大越先匹配。自定义任务为 100，通用技能 50，兜底类技能更低。"""
    timeout: ClassVar[float | None] = 30.0
    """单次 handle 的超时（秒），None 表示不限。长任务请在 handle 中用 ctx.spawn 放到后台。"""
    task_kind: ClassVar[TaskKind] = TaskKind.FOREGROUND
    """handle 以何种任务类型运行。"""

    def __init__(self) -> None:
        self._compiled: list[tuple[str, re.Pattern[str]]] = [
            (intent, re.compile(p)) for intent, plist in self.patterns.items() for p in plist
        ]

    # ------------------------------------------------------------------ 生命周期

    async def setup(self, ctx: SkillContext) -> None:  # noqa: B027 - 可选钩子
        """技能加载时调用一次（初始化后台资源，例如调度器、播放器）。"""

    async def teardown(self) -> None:  # noqa: B027 - 可选钩子
        """程序退出时调用（释放资源）。"""

    # ------------------------------------------------------------------ 匹配与处理

    def match(self, text: str) -> Intent | None:
        """用规则匹配归一化文本，命中返回意图，否则返回 None。"""
        for intent_name, pattern in self._compiled:
            m = pattern.search(text)
            if m:
                slots = {k: v for k, v in m.groupdict().items() if v is not None}
                return Intent(skill=self.name, name=intent_name, text=text, slots=slots)
        return None

    @abstractmethod
    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        """处理意图。

        运行在 TaskManager 托管的任务中，可被打断 / “停止”取消（抛出 CancelledError），
        请把需要清理的资源放在 ``try/finally`` 中。
        """

    def tools(self) -> list[ToolSpec]:
        """暴露给 AI Agent 的工具列表，默认不暴露。"""
        return []
