"""SkillContext：技能运行环境（技能唯一可以依赖的对象）。

为什么技能不能直接 import 其他模块？
    - 解耦：底层（音频、对话、任务管理）重构不影响技能；
    - 可测试：单元测试时传入一个假的上下文即可；
    - 统一治理：技能启动的后台任务必须经过 TaskManager，才能被“停止”“取消XX”终止。

每个技能拿到的是自己专属的上下文（``settings`` 和 ``data_dir`` 互相隔离）。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
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
        tools: Mapping[str, Any] | None = None,
    ) -> None:
        self.skill_name = skill_name
        self.config = config
        self.tasks = tasks
        self._bus = bus
        self._dialog = dialog
        self._output = output
        # 与 App 共用同一个字典。技能加载时 MCP 还没连上，稍后由 App 填进去。
        self._tools = tools if tools is not None else {}
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

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """调用已经加载的工具（通常是 MCP），把结果收成一段文本。

        工具要在调用时已经放进构造时传入的字典。没加载会抛出 ``LookupError``。
        可取消：工具内部的 ``CancelledError`` 会原样传出，不被收成错误文本。

        Args:
            name: 工具全名，例如 ``music-mcp-server_get_audio_url``。
            args: 工具参数。

        Returns:
            工具返回的文本。内容块列表会拼在一起；``(文本, 附件)`` 只取文本。

        Raises:
            LookupError: 这个名字的工具没有加载。
        """
        tool = self._tools.get(name)
        if tool is None:
            raise LookupError(name)
        coroutine = getattr(tool, "coroutine", None)
        result = await coroutine(**args) if coroutine is not None else await tool.ainvoke(args)
        return _tool_text(result)


def _tool_text(result: Any) -> str:
    """把工具的各种返回值收成一段文本。"""
    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(result)
