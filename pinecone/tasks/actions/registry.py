"""动作注册表：``type`` 名 → 动作类。"""

from __future__ import annotations

from typing import TypeVar

from pydantic import ValidationError

from .base import Action

A = TypeVar("A", bound=type[Action])

_ACTIONS: dict[str, type[Action]] = {}


def register_action(cls: A) -> A:
    """类装饰器：注册动作类型。重复注册同名类型时后者覆盖前者（便于插件替换内置实现）。"""
    if not cls.type:
        raise ValueError(f"{cls.__name__} 没有声明 type")
    _ACTIONS[cls.type] = cls
    return cls


def registered_actions() -> dict[str, type[Action]]:
    """全部已注册的动作类型。"""
    return dict(_ACTIONS)


def create_action(type_: str, params: dict) -> Action:
    """按类型创建动作并校验参数。

    Raises:
        ValueError: 未知类型或参数不合法（加载任务时即报错，而不是运行时才发现）。
    """
    cls = _ACTIONS.get(type_)
    if cls is None:
        raise ValueError(f"未知的动作类型 {type_!r}，可用：{', '.join(sorted(_ACTIONS))}")
    try:
        return cls(cls.Params.model_validate(params))
    except ValidationError as exc:
        raise ValueError(f"动作 {type_} 参数错误：{exc}") from exc
