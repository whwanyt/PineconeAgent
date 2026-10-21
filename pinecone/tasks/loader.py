"""加载 ``tasks/*.yaml``。

- 每个文件一个任务，文件名以 ``_`` 开头的会被忽略（可用来临时禁用任务）；
- 字符串中的 ``${ENV}`` / ``${ENV:默认值}`` 会展开为环境变量，密钥不要直接写在 YAML 里；
- 加载时就创建一遍所有动作来校验参数，错误的任务记录日志后跳过，不影响其他任务和主程序；
- 任务目录会加入 ``sys.path``，python 动作可以直接引用放在 ``tasks/`` 里的脚本模块。
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from pinecone.core.config import expand_env

from .actions import create_action
from .model import TaskSpec

logger = logging.getLogger(__name__)


class TaskLoadError(ValueError):
    """任务文件不合法。"""


def load_task_file(path: Path) -> TaskSpec:
    """加载并校验单个任务文件。

    Raises:
        TaskLoadError: YAML 语法、字段、正则或动作参数错误。
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = TaskSpec.model_validate(expand_env(raw))
    except (yaml.YAMLError, ValidationError) as exc:
        raise TaskLoadError(f"{path.name}：{exc}") from exc
    spec.source = path

    for pattern in spec.patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise TaskLoadError(f"{path.name}：正则 {pattern!r} 不合法：{exc}") from exc
    for action in [*spec.actions, *spec.on_cancel, *spec.on_error]:
        try:
            create_action(action.type, action.params)
        except ValueError as exc:
            raise TaskLoadError(f"{path.name}：{exc}") from exc
    return spec


def load_tasks(dirs: list[Path]) -> list[TaskSpec]:
    """加载多个目录中的全部任务，跳过不合法的文件和重名任务。"""
    tasks: dict[str, TaskSpec] = {}
    for directory in dirs:
        if not directory.is_dir():
            logger.debug("任务目录不存在：%s", directory)
            continue
        if str(directory) not in sys.path:
            sys.path.append(str(directory))
        for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
            if path.name.startswith("_"):
                continue
            try:
                spec = load_task_file(path)
            except TaskLoadError as exc:
                logger.error("跳过任务 %s", exc)
                continue
            if spec.name in tasks:
                logger.error(
                    "跳过任务 %s：与 %s 重名「%s」", path.name, tasks[spec.name].source, spec.name
                )
                continue
            tasks[spec.name] = spec
    logger.info("已加载 %d 个自定义任务：%s", len(tasks), "、".join(tasks) or "无")
    return list(tasks.values())
