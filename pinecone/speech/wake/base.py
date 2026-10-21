"""唤醒词检测接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WakeWordDetector(ABC):
    """唤醒词检测器。

    线程模型：:meth:`accept_frame` 在采集工作线程中被逐帧调用；
    :meth:`set_threshold_boost` 可以在任意线程调用（实现需保证线程安全）。
    """

    @abstractmethod
    def accept_frame(self, frame: np.ndarray) -> str | None:
        """输入一帧 16kHz float32 音频。

        Returns:
            检测到唤醒词时返回唤醒词文本，否则返回 ``None``。
        """

    @abstractmethod
    def set_threshold_boost(self, boost: float) -> None:
        """临时提高触发阈值（播报期间使用，防止被自己的声音唤醒）。``0`` 表示恢复正常。"""

    @abstractmethod
    def reset(self) -> None:
        """清空内部状态（例如刚触发过一次唤醒后）。"""
