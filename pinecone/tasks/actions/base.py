"""动作基类与执行上下文。

取消约定（所有动作都必须遵守）：
    - 动作运行在 TaskManager 托管的任务中，取消时 ``await`` 处会抛出 ``CancelledError``；
    - 持有外部资源（子进程、网络连接、线程）的动作，必须在 ``finally`` 或
      ``ctx.token.on_cancel`` 回调中释放，确保“取消XX”之后真的停下来。
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel

from pinecone.core.cancel import CancelToken

_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render(value: Any, variables: dict[str, Any]) -> Any:
    """把 ``value`` 中的 ``{{变量}}`` 替换为变量值（递归处理 dict / list）。

    用双花括号而不是 Python 的 ``{}``，是因为 shell 命令和 JSON 里经常出现单花括号。
    未定义的变量替换为空字符串。
    """
    if isinstance(value, str):
        return _VAR.sub(lambda m: str(variables.get(m.group(1), "")), value)
    if isinstance(value, dict):
        return {k: render(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, variables) for v in value]
    return value


class ActionError(RuntimeError):
    """动作执行失败（例如命令返回非 0、HTTP 状态码 >= 400）。"""


@dataclass
class ActionContext:
    """动作执行上下文。

    Attributes:
        task_name: 所属任务名。
        variables: 可在参数中用 ``{{变量名}}`` 引用的变量：
            ``text``（用户原话）、正则命名分组、``output``（上一个动作的输出）、
            ``reason``（仅 on_cancel 中：取消原因）、``error``（仅 on_error 中）。
        say: 播报函数（speak 动作使用）。
        token: 当前任务的取消令牌。
        logger: 日志记录器。
    """

    task_name: str
    say: Callable[[str], Awaitable[None]]
    token: CancelToken
    variables: dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("pinecone.tasks"))

    def render(self, value: Any) -> Any:
        """用当前变量替换参数中的 ``{{变量}}``，见 :func:`render`。"""
        return render(value, self.variables)


class Action(ABC):
    """动作基类。子类声明 ``type`` 和参数模型 ``Params``，实现 :meth:`run`。"""

    type: ClassVar[str] = ""
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: BaseModel) -> None:
        self.params = params

    @abstractmethod
    async def run(self, ctx: ActionContext) -> str | None:
        """执行动作，返回输出文本（会作为 ``{{output}}`` 传给下一个动作）。

        Raises:
            ActionError: 执行失败。
            asyncio.CancelledError: 被取消。
        """
