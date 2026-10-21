"""python 动作：调用一个 Python 函数。

YAML 示例::

    - type: python
      callable: "my_tasks.backup:run"   # 模块:函数；模块从项目根目录和 tasks/ 目录导入
      args: {target: "/mnt/backup"}     # 关键字参数，支持 {{变量}}

函数约定：
    - 可以是普通函数或 async 函数，返回值转成字符串作为 ``{{output}}``；
    - 如果函数有名为 ``token`` 的参数，会注入 :class:`CancelToken`；
    - 普通函数在线程池中执行，**线程无法被强制终止**，长时间运行的函数必须在循环中
      检查 ``token.cancelled`` 或调用 ``token.raise_if_cancelled()``，否则取消后线程仍会跑完
      （任务本身会立即结束，但副作用会继续）。
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, field_validator

from .base import Action, ActionContext
from .registry import register_action


def load_callable(spec: str) -> Callable[..., Any]:
    """按 ``模块:函数`` 导入函数。"""
    module_name, _, func_name = spec.partition(":")
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ValueError(f"{spec} 不是可调用对象")
    return func


@register_action
class PythonAction(Action):
    type = "python"

    class Params(BaseModel):
        callable: str
        args: dict[str, Any] = {}

        @field_validator("callable")
        @classmethod
        def _format(cls, value: str) -> str:
            if ":" not in value:
                raise ValueError("格式应为 模块:函数，例如 my_tasks.backup:run")
            return value

    params: Params

    async def run(self, ctx: ActionContext) -> str | None:
        func = load_callable(self.params.callable)
        kwargs = ctx.render(self.params.args)
        if "token" in inspect.signature(func).parameters:
            kwargs["token"] = ctx.token
        ctx.logger.info("[%s] python：%s", ctx.task_name, self.params.callable)
        if inspect.iscoroutinefunction(func):
            result = await func(**kwargs)
        else:
            result = await asyncio.to_thread(func, **kwargs)
        return None if result is None else str(result)
