"""任务 YAML 的数据结构。

一个文件一个任务，完整字段见 ``tasks/README.md`` 和示例文件。动作的具体参数由各动作类型
自己的 ``Params`` 模型校验（见 :mod:`pinecone.tasks.actions`），这里只要求有 ``type`` 字段。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionSpec(BaseModel):
    """一个动作：``type`` 决定动作类型，其余字段是该类型的参数。"""

    model_config = ConfigDict(extra="allow")

    type: str

    @property
    def params(self) -> dict[str, Any]:
        """除 ``type`` 外的全部参数。"""
        return dict(self.model_extra or {})


class TaskSpec(BaseModel):
    """一个自定义任务。"""

    name: str = Field(description="任务名（中文），用于“取消XX”和“现在在做什么”的播报")
    description: str = Field(default="", description="任务说明，AI 据此判断什么时候调用")
    keywords: list[str] = Field(
        default_factory=list, description="触发说法，命中其中任一即可（包含匹配）"
    )
    patterns: list[str] = Field(
        default_factory=list,
        description="正则触发，命名分组会作为变量传给动作，例如 (?P<room>客厅|卧室)",
    )
    background: bool = Field(default=False, description="是否后台执行（不阻塞对话，唤醒不会打断）")
    timeout: float | None = Field(default=30, description="超时秒数，超时自动取消；null 表示不限")
    cancellable: bool = Field(default=True, description="是否允许被语音 / AI 取消")
    reply: str | None = Field(
        default=None,
        description="后台任务启动时的即时回复；前台任务默认不回复（由 speak 动作播报）",
    )
    done_reply: str | None = Field(default=None, description="后台任务完成后的主动播报，支持变量")
    expose_to_agent: bool = Field(default=True, description="是否作为工具暴露给 AI")
    actions: list[ActionSpec] = Field(min_length=1)
    on_cancel: list[ActionSpec] = Field(
        default_factory=list, description="被取消 / 超时后执行的清理动作"
    )
    on_error: list[ActionSpec] = Field(default_factory=list, description="执行失败后执行的动作")

    source: Path | None = Field(default=None, exclude=True)
    """来源文件（加载器填入，便于报错定位）。"""

    @model_validator(mode="after")
    def _need_trigger(self) -> TaskSpec:
        if not self.keywords and not self.patterns and not self.expose_to_agent:
            raise ValueError("keywords / patterns 至少配置一个，或者开启 expose_to_agent")
        return self
