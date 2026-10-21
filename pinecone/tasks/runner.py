"""任务执行器：按顺序执行一个任务的动作。

执行器本身不关心前台 / 后台、超时和取消——这些由调用方（custom_task 技能）通过
TaskManager 托管来实现。执行器只负责：

1. 依次创建并执行动作，上一个动作的输出作为 ``{{output}}`` 传给下一个；
2. 动作失败时执行 ``on_error`` 动作（变量 ``{{error}}``），然后重新抛出异常；
3. 提供 :meth:`run_cleanup` 给 ``on_cancel`` 使用（变量 ``{{reason}}``）。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pinecone.core.cancel import CancelToken, current_token

from .actions import ActionContext, create_action
from .model import ActionSpec, TaskSpec

logger = logging.getLogger(__name__)

# 播报取消原因时使用的中文
REASON_TEXT = {"timeout": "超时", "user": "用户取消", "barge_in": "被打断", "shutdown": "程序退出"}


class TaskRunner:
    """执行自定义任务的动作序列。"""

    def __init__(self, say: Callable[[str], Awaitable[None]]) -> None:
        self._say = say

    async def run(self, spec: TaskSpec, variables: dict[str, Any] | None = None) -> str | None:
        """执行任务的全部动作，返回最后一个动作的输出。

        必须在 TaskManager 托管的任务中调用（取消令牌从上下文获取）。

        Raises:
            Exception: 任一动作失败（on_error 动作执行完后重新抛出）。
            asyncio.CancelledError: 被取消。
        """
        ctx = self._context(spec, current_token(), variables)
        try:
            return await self._run_actions(spec.actions, ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if spec.on_error:
                ctx.variables["error"] = str(exc)
                try:
                    await self._run_actions(spec.on_error, ctx)
                except Exception:
                    logger.exception("任务「%s」的 on_error 动作失败", spec.name)
            raise

    async def run_cleanup(
        self, spec: TaskSpec, reason: str, variables: dict[str, Any] | None = None
    ) -> None:
        """执行 on_cancel 动作（任务被取消或超时后调用）。"""
        if not spec.on_cancel:
            return
        ctx = self._context(spec, CancelToken(), variables)
        ctx.variables["reason"] = REASON_TEXT.get(reason, reason)
        await self._run_actions(spec.on_cancel, ctx)

    def _context(
        self, spec: TaskSpec, token: CancelToken, variables: dict[str, Any] | None
    ) -> ActionContext:
        return ActionContext(
            task_name=spec.name,
            say=self._say,
            token=token,
            variables=dict(variables or {}),
            logger=logging.getLogger(f"pinecone.task.{spec.name}"),
        )

    @staticmethod
    async def _run_actions(actions: list[ActionSpec], ctx: ActionContext) -> str | None:
        output: str | None = None
        for spec in actions:
            action = create_action(spec.type, spec.params)
            result = await action.run(ctx)
            if result is not None:
                output = result
                ctx.variables["output"] = result
        return output
