"""speak 动作：播报一句话。

YAML 示例::

    - type: speak
      text: "备份完成：{{output}}"

取消：播报本身可被打断 / 取消，取消时立即停止发声。
"""

from __future__ import annotations

from pydantic import BaseModel

from .base import Action, ActionContext
from .registry import register_action


@register_action
class SpeakAction(Action):
    type = "speak"

    class Params(BaseModel):
        text: str

    params: Params

    async def run(self, ctx: ActionContext) -> str | None:
        text = ctx.render(self.params.text).strip()
        if text:
            await ctx.say(text)
        return None
