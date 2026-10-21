"""语音活动检测接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpeechStart:
    """检测到开始说话。"""


@dataclass
class SpeechEnd:
    """一句话说完。``audio`` 为整句 16kHz float32 音频（含少量前后余量）。"""

    audio: np.ndarray = field(repr=False)


VadEvent = SpeechStart | SpeechEnd


class VoiceActivityDetector(ABC):
    """语音活动检测器。

    线程模型：所有方法都只在采集工作线程中调用，实现无需考虑线程安全。
    """

    @abstractmethod
    def accept_frame(self, frame: np.ndarray) -> list[VadEvent]:
        """输入一帧音频，返回本帧产生的事件（通常为空列表）。"""

    @abstractmethod
    def reset(self) -> None:
        """清空内部状态和缓冲（开始新一轮聆听时调用）。"""

    @property
    @abstractmethod
    def in_speech(self) -> bool:
        """当前是否处于说话中。"""
