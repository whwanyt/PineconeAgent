"""技能框架。

- :mod:`base`：技能基类 :class:`Skill`、意图 :class:`Intent`、结果 :class:`SkillResult`、
  暴露给 AI 的工具描述 :class:`ToolSpec`；
- :mod:`context`：技能唯一可依赖的运行环境 :class:`SkillContext`；
- :mod:`registry`：``@register_skill`` 注册、内置技能与 ``plugins/`` 插件自动发现、
  按配置实例化与生命周期管理（:class:`SkillManager`）；
- ``builtin/``：内置技能，每个技能一个独立包；
- ``_template/``：新增技能的模板（复制到 ``plugins/`` 下修改即可）。
"""

from pinecone.skills.base import Intent, Skill, SkillResult, ToolSpec
from pinecone.skills.context import SkillContext
from pinecone.skills.registry import SkillManager, register_skill

__all__ = [
    "Intent",
    "Skill",
    "SkillContext",
    "SkillManager",
    "SkillResult",
    "ToolSpec",
    "register_skill",
]
