"""技能注册、自动发现与生命周期管理。

新增技能的三种方式（都不需要修改主程序代码）：
    1. 内置技能：在 ``pinecone/skills/builtin/`` 下新建一个包；
    2. 插件技能：把 ``pinecone/skills/_template/`` 复制到 ``plugins/<技能目录>/``；
    3. 在 config.yaml 的 ``skills.plugin_dirs`` 中添加其他插件目录。
包中任意模块里用 ``@register_skill`` 装饰的类都会被注册（包的 ``__init__.py`` 需要导入它）。

以下划线开头的目录（如 ``_template``）不会被加载。
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pkgutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pinecone.core.config import AppConfig
from pinecone.core.task_manager import TaskManager, TaskStatus
from pinecone.skills.base import Intent, Skill, SkillResult
from pinecone.skills.context import SkillContext

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[Skill]] = {}
S = TypeVar("S", bound=type[Skill])


def register_skill(cls: S) -> S:
    """类装饰器：注册技能。

    Raises:
        ValueError: 技能没有 name，或与已注册的其他技能重名。
    """
    if not cls.name:
        raise ValueError(f"技能 {cls.__qualname__} 必须声明 name")
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing.__qualname__ != cls.__qualname__:
        raise ValueError(f"技能名重复：{cls.name}（{existing.__module__} 与 {cls.__module__}）")
    _REGISTRY[cls.name] = cls
    return cls


def registered_skills() -> dict[str, type[Skill]]:
    """返回所有已注册的技能类（名称 → 类）。"""
    return dict(_REGISTRY)


def discover_builtin() -> None:
    """导入 ``pinecone.skills.builtin`` 下的全部技能包，触发注册。"""
    import pinecone.skills.builtin as builtin

    for module in pkgutil.iter_modules(builtin.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{builtin.__name__}.{module.name}")


def discover_plugins(plugin_dir: Path) -> None:
    """导入插件目录下的每个技能包。

    每个包以 ``pinecone_plugins.<目录名>`` 的模块名导入，因此包内可以使用相对导入
    （``from .helper import xxx``）。单个插件导入失败只记录日志，不影响其他技能。
    """
    if not plugin_dir.is_dir():
        return
    for pkg in sorted(plugin_dir.iterdir()):
        init = pkg / "__init__.py"
        if pkg.name.startswith(("_", ".")) or not init.exists():
            continue
        module_name = f"pinecone_plugins.{pkg.name}"
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, init, submodule_search_locations=[str(pkg)]
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logger.info("已加载插件目录：%s", pkg)
        except Exception:
            sys.modules.pop(module_name, None)
            logger.exception("插件加载失败：%s", pkg)


ContextFactory = Callable[[str], SkillContext]


class SkillManager:
    """按配置实例化技能、管理生命周期、执行规则匹配与技能调用。"""

    def __init__(
        self, config: AppConfig, tasks: TaskManager, context_factory: ContextFactory
    ) -> None:
        self._config = config
        self._tasks = tasks
        self._context_factory = context_factory
        self._skills: dict[str, Skill] = {}
        self._contexts: dict[str, SkillContext] = {}

    @property
    def skills(self) -> list[Skill]:
        """已启用的技能（按优先级从高到低）。"""
        return sorted(self._skills.values(), key=lambda s: -s.priority)

    def get(self, name: str) -> Skill | None:
        """按名称获取已启用的技能。"""
        return self._skills.get(name)

    def context(self, name: str) -> SkillContext:
        """获取技能的上下文。"""
        return self._contexts[name]

    async def load(self) -> None:
        """发现并实例化启用的技能，依次调用 setup。单个技能 setup 失败会被禁用。"""
        discover_builtin()
        for directory in self._config.skills.plugin_dirs:
            discover_plugins(self._config.resolve_path(directory))

        cfg = self._config.skills
        for name, cls in registered_skills().items():
            enabled = "*" in cfg.enabled or name in cfg.enabled
            if not enabled or name in cfg.disabled:
                continue
            ctx = self._context_factory(name)
            try:
                skill = cls()
                await skill.setup(ctx)
            except Exception:
                logger.exception("技能 %s 初始化失败，已禁用", name)
                continue
            self._skills[name] = skill
            self._contexts[name] = ctx
        logger.info("已加载技能：%s", "、".join(s.name for s in self.skills) or "无")

    async def unload(self) -> None:
        """依次调用 teardown。"""
        for skill in self.skills:
            try:
                await skill.teardown()
            except Exception:
                logger.exception("技能 %s 释放资源失败", skill.name)
        self._skills.clear()

    def match(self, text: str) -> tuple[Skill, Intent] | None:
        """按优先级规则匹配，返回第一个命中的技能与意图。"""
        for skill in self.skills:
            intent = skill.match(text)
            if intent is not None:
                return skill, intent
        return None

    async def execute(self, skill: Skill, intent: Intent) -> SkillResult:
        """在托管任务中执行技能（可在任务列表中看到、可被取消、受超时约束）。

        Returns:
            技能结果；超时 / 失败 / 被取消时返回对应的提示。

        Raises:
            asyncio.CancelledError: 调用方（对话任务）被取消时向上传播。
        """
        ctx = self._contexts[skill.name]
        rt = await self._tasks.run(
            skill.handle(ctx, intent),
            name=f"{skill.description or skill.name}",
            kind=skill.task_kind,
            source=f"skill:{skill.name}",
            timeout=skill.timeout,
        )
        if rt.status == TaskStatus.DONE:
            return rt.result if isinstance(rt.result, SkillResult) else SkillResult()
        if rt.status == TaskStatus.TIMEOUT:
            return SkillResult(reply="操作超时了，请稍后再试。")
        if rt.status == TaskStatus.CANCELLED:
            return SkillResult()
        return SkillResult(reply="抱歉，执行时出了点问题。")
