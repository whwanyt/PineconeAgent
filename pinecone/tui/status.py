"""状态栏：助手当前状态、追问倒计时、正在运行的任务。"""

from __future__ import annotations

from typing import Protocol

from rich.text import Text
from textual.widgets import Static

from pinecone.core.state import AssistantState as S
from pinecone.core.state import StateMachine
from pinecone.core.task_manager import TaskManager


class StatusView(Protocol):
    """状态栏要读的那一小块会话表面，不依赖对话控制器。"""

    state: StateMachine
    tasks: TaskManager

    @property
    def timer_remaining(self) -> float | None:
        """聆听或追问还剩几秒，没有在计时时为 None。"""
        ...


_STATES = {
    S.IDLE: ("空闲", "dim"),
    S.LISTENING: ("聆听中", "bold cyan"),
    S.THINKING: ("思考中", "bold yellow"),
    S.SPEAKING: ("播报中", "bold green"),
    S.FOLLOW_UP: ("等待追问", "bold magenta"),
}

_REFRESH_S = 0.25


class StatusBar(Static):
    """状态栏。定时刷新，倒计时和任务列表不用逐个订阅事件。

    任务只列技能、自定义任务、AI 这些；「对话」「提示音」这类对话流程自己的任务不列。
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__("启动中……", id=id)
        self._source: StatusView | None = None

    def watch(self, source: StatusView) -> None:
        """后端启动完成后调用，开始显示。"""
        self._source = source
        self.refresh_status()
        self.set_interval(_REFRESH_S, self.refresh_status)

    def refresh_status(self) -> None:
        if self._source is None:
            return
        source = self._source
        label, style = _STATES[source.state.state]
        line = Text()
        line.append("● ", style=style)
        line.append(label, style=style)

        remaining = source.timer_remaining
        if remaining is not None and source.state.state in (S.LISTENING, S.FOLLOW_UP):
            line.append(f"  {remaining:.0f}s", style="dim")

        names = [t.name for t in source.tasks.list() if t.source != "dialog"]
        if names:
            line.append("   任务：", style="dim")
            line.append("、".join(names))
        self.update(line)
