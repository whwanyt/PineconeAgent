"""sleep 动作：等待一段时间（例如“开灯 → 等 10 分钟 → 关灯”）。

YAML 示例::

    - type: sleep
      seconds: 600

取消：``asyncio.sleep`` 可以立即被取消。
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from .base import Action, ActionContext
from .registry import register_action


@register_action
class SleepAction(Action):
    type = "sleep"

    class Params(BaseModel):
        seconds: float = Field(ge=0)

    params: Params

    async def run(self, ctx: ActionContext) -> str | None:
        await asyncio.sleep(self.params.seconds)
        return None
