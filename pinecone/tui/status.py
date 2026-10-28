"""状态栏：助手当前状态、追问倒计时、正在运行的任务。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Static

from pinecone.core.state import AssistantState as S

if TYPE_CHECKING:
    from pinecone.core.state import StateMachine
    from pinecone.core.task_manager import TaskManager
    from pinecone.dialog import DialogController

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
        self._source: tuple[StateMachine, DialogController, TaskManager] | None = None

    def watch(self, state: StateMachine, dialog: DialogController, tasks: TaskManager) -> None:
        """后端启动完成后调用，开始显示。"""
        self._source = (state, dialog, tasks)
        self.refresh_status()
        self.set_interval(_REFRESH_S, self.refresh_status)

    def refresh_status(self) -> None:
        if self._source is None:
            return
        state, dialog, tasks = self._source
        label, style = _STATES[state.state]
        line = Text()
        line.append("● ", style=style)
        line.append(label, style=style)

        remaining = dialog.timer_remaining
        if remaining is not None and state.state in (S.LISTENING, S.FOLLOW_UP):
            line.append(f"  {remaining:.0f}s", style="dim")

        names = [t.name for t in tasks.list() if t.source != "dialog"]
        if names:
            line.append("   任务：", style="dim")
            line.append("、".join(names))
        self.update(line)
