"""测试公共夹具：用假的对话控制器和输出构造技能运行环境。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pinecone.core.bus import EventBus
from pinecone.core.config import AppConfig
from pinecone.core.task_manager import TaskKind, TaskManager
from pinecone.skills import SkillContext, SkillManager


class FakeDialog:
    """记录播报内容的假对话控制器。"""

    def __init__(self, tasks: TaskManager) -> None:
        self.tasks = tasks
        self.spoken: list[str] = []
        self.interrupted = 0

    async def speak(self, text: str) -> None:
        self.spoken.append(text)

    def announce(self, text: str, *, name: str = "播报", kind: TaskKind = TaskKind.FOREGROUND):
        return self.tasks.spawn(self.speak(text), name=name, kind=kind)

    async def interrupt(self, reason: str = "user") -> int:
        self.interrupted += 1
        return self.tasks.cancel_foreground(reason)


class FakeOutput:
    def __init__(self) -> None:
        self.played = 0

    async def play(self, samples, rate) -> None:
        self.played += 1
        await asyncio.sleep(0.01)


class SkillEnv:
    """测试用技能环境：配置 + 任务管理 + 假对话 + 技能管理器。"""

    def __init__(self, tmp_path: Path) -> None:
        self.config = AppConfig(data_dir=str(tmp_path / "data"))
        self.config.skills.plugin_dirs = []
        self.bus = EventBus()
        self.tasks = TaskManager(self.bus)
        self.dialog = FakeDialog(self.tasks)
        self.output = FakeOutput()
        self.manager = SkillManager(self.config, self.tasks, self.make_context)

    def make_context(self, name: str) -> SkillContext:
        ctx = SkillContext(
            skill_name=name,
            config=self.config,
            bus=self.bus,
            tasks=self.tasks,
            dialog=self.dialog,  # type: ignore[arg-type]
            output=self.output,  # type: ignore[arg-type]
        )
        ctx.extras["skills"] = self.manager
        return ctx


@pytest.fixture
def skill_env(tmp_path: Path) -> SkillEnv:
    return SkillEnv(tmp_path)
