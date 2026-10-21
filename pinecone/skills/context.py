"""SkillContext：技能运行环境（技能唯一可以依赖的对象）。

为什么技能不能直接 import 其他模块？
    - 解耦：底层（音频、对话、任务管理）重构不影响技能；
    - 可测试：单元测试时传入一个假的上下文即可；
    - 统一治理：技能启动的后台任务必须经过 TaskManager，才能被“停止”“取消XX”终止。

每个技能拿到的是自己专属的上下文（``settings`` 和 ``data_dir`` 互相隔离）。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pinecone.core.events import StateChanged
from pinecone.core.task_manager import OnCancel, RunningTask, TaskKind, TaskTarget

if TYPE_CHECKING:
    from pinecone.audio.output import AudioOutput
    from pinecone.core.bus import EventBus
    from pinecone.core.config import AppConfig
    from pinecone.core.task_manager import TaskManager
    from pinecone.dialog import DialogController


class SkillContext:
    """技能运行环境。"""

    def __init__(
        self,
        *,
        skill_name: str,
        config: AppConfig,
        bus: EventBus,
        tasks: TaskManager,
        dialog: DialogController,
        output: AudioOutput,
    ) -> None:
        self.skill_name = skill_name
        self.config = config
        self.tasks = tasks
        self._bus = bus
        self._dialog = dialog
        self._output = output
        self.logger = logging.getLogger(f"pinecone.skill.{skill_name}")
        self.extras: dict[str, Any] = {}
        """共享扩展点：由应用注入的额外服务（例如技能管理器、自定义任务引擎）。"""

    # ------------------------------------------------------------------ 配置与存储

    @property
    def settings(self) -> dict[str, Any]:
        """本技能的私有配置（config.yaml 中 ``skills.settings.<技能名>``）。"""
        return self.config.skills.settings.get(self.skill_name, {})

    @property
    def data_dir(self) -> Path:
        """本技能的数据目录（``data/<技能名>/``，自动创建）。"""
        path = self.config.data_path / self.skill_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------ 语音

    async def say(self, text: str) -> None:
        """主动播报（带状态管理，见 DialogController.speak）。可取消。"""
        await self._dialog.speak(text)

    def announce(
        self, text: str, *, name: str = "播报", kind: TaskKind = TaskKind.FOREGROUND
    ) -> RunningTask:
        """在托管任务中主动播报，不等待播完。"""
        return self._dialog.announce(text, name=name, kind=kind)

    async def play_sound(self, samples: np.ndarray, sample_rate: int) -> None:
        """播放一段音频（例如闹钟铃声）。可取消：取消时立即停止。"""
        await self._output.play(samples, sample_rate)

    async def interrupt(self, reason: str = "user") -> int:
        """停止当前播报和所有前台任务（调用方自身除外），返回取消数量。"""
        return await self._dialog.interrupt(reason)

    # ------------------------------------------------------------------ 任务

    def spawn(
        self,
        target: TaskTarget,
        *,
        name: str,
        kind: TaskKind = TaskKind.BACKGROUND,
        timeout: float | None = None,
        cancellable: bool = True,
        on_cancel: OnCancel | None = None,
    ) -> RunningTask:
        """启动一个托管任务（默认后台），可被“停止所有任务”“取消XX”终止。"""
        return self.tasks.spawn(
            target,
            name=name,
            kind=kind,
            source=f"skill:{self.skill_name}",
            timeout=timeout,
            cancellable=cancellable,
            on_cancel=on_cancel,
        )

    # ------------------------------------------------------------------ 事件

    def on_state_change(
        self, callback: Callable[[str, str], Awaitable[None] | None]
    ) -> Callable[[], None]:
        """订阅助手状态变化（参数：旧状态、新状态），返回取消订阅函数。

        例如音乐技能据此在唤醒时压低音量、对话结束后恢复。
        """
        return self._bus.subscribe(StateChanged, lambda e: callback(e.old, e.new))
