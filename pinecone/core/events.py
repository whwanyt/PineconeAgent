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


# ---- 播报 ----


@dataclass(frozen=True, kw_only=True)
class Spoken(Event):
    """开始播报一句话（已合成、送入扬声器）。

    ``utterance`` 是本次播报的编号，同一次 ``say`` / ``say_stream`` 的各句相同，
    界面可以据此把流式回复的多句拼成一条。
    """

    text: str
    utterance: int


@dataclass(frozen=True, kw_only=True)
class ChoicesShown(Event):
    """回复附带的编号候选列表（语音模式只念问句，文本界面显示这份列表）。"""

    lines: str


@dataclass(frozen=True, kw_only=True)
class PlaylistSnapshot:
    """某一时刻的播放队列。

    界面启动时先读这一帧，之后只听 :class:`PlaylistUpdated`。
    ``current`` 从 1 数，没有正在播放的曲目时为 0。
    """

    tracks: tuple[str, ...] = ()
    current: int = 0
    mode: str = ""
    paused: bool = False


@dataclass(frozen=True, kw_only=True)
class PlaylistUpdated(Event):
    """当前播放队列发生了变化。

    只在曲目、当前位置、播放模式或暂停状态真正变化时发布，供界面刷新。
    点选切歌不走这条事件的反方向：界面通过会话门面发命令，技能改完队列后再发布。

    ``tracks`` 是按播放顺序排列的曲目名。``current`` 从 1 数，没有正在播放的曲目时为 0。
    """

    tracks: tuple[str, ...]
    current: int
    mode: str
    paused: bool


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
