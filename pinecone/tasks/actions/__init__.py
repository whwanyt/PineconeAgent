"""动作类型。

内置：shell / http / python / speak / sleep。新增动作类型只需在本目录加一个文件::

    from pydantic import BaseModel
    from pinecone.tasks.actions.base import Action, ActionContext
    from pinecone.tasks.actions.registry import register_action

    @register_action
    class MqttAction(Action):
        type = "mqtt"

        class Params(BaseModel):
            topic: str
            payload: str = ""

        async def run(self, ctx: ActionContext) -> str | None:
            ...

并在下面的导入列表中加上它（或放在插件技能里导入，同样会注册）。
"""

from . import http, python, shell, sleep, speak  # noqa: F401 - 导入即注册
from .base import Action, ActionContext, ActionError, render
from .registry import create_action, register_action, registered_actions

__all__ = [
    "render",
    "Action",
    "ActionContext",
    "ActionError",
    "create_action",
    "register_action",
    "registered_actions",
]
