"""事件定义。

所有在事件总线上流转的事件都定义在这里，便于统一查阅“系统里会发生哪些事”。
事件都是不可变的 dataclass，订阅者只能读取不能修改，避免多个订阅者之间互相影响。

命名约定：用过去式或名词描述“已经发生的事情”，例如 ``WakeWordDetected``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, kw_only=True)
class Event:
    """所有事件的基类。"""

    timestamp: float = field(default_factory=time.monotonic)
    """事件产生时的单调时钟时间（秒），用于计算延迟，不受系统时间调整影响。"""


# ---- 状态机 ----


@dataclass(frozen=True, kw_only=True)
class StateChanged(Event):
    """助手状态发生变化。``old`` / ``new`` 为 :class:`AssistantState` 的值。"""

    old: str
    new: str
    reason: str = ""


# ---- 语音链路 ----


@dataclass(frozen=True, kw_only=True)
class WakeWordDetected(Event):
    """检测到唤醒词。``keyword`` 为命中的唤醒词文本。"""

    keyword: str


@dataclass(frozen=True, kw_only=True)
class SpeechStarted(Event):
    """VAD 检测到用户开始说话。"""


@dataclass(frozen=True, kw_only=True)
class SpeechEnded(Event):
    """VAD 检测到一段完整语音结束。``audio`` 为 16kHz float32 单声道样本。"""

    audio: np.ndarray = field(repr=False)


@dataclass(frozen=True, kw_only=True)
class Transcribed(Event):
    """语音识别完成。"""

    text: str


@dataclass(frozen=True, kw_only=True)
class BargeIn(Event):
    """发生了对话打断。``source`` 为 ``wake_word`` 或 ``vad``。"""

    source: str


# ---- 任务管理 ----


@dataclass(frozen=True, kw_only=True)
class TaskStarted(Event):
    """托管任务开始运行。"""

    task_id: str
    name: str
    kind: str


@dataclass(frozen=True, kw_only=True)
class TaskFinished(Event):
    """托管任务结束。``status`` 取值：``done`` / ``cancelled`` / ``timeout`` / ``failed``。"""

    task_id: str
    name: str
    kind: str
    status: str
    error: str | None = None
    result: Any = field(default=None, repr=False)
