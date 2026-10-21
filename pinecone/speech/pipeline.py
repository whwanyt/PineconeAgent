"""前端语音处理流水线：把麦克风帧送给唤醒词和 VAD，并把结果发布为事件。

运行在采集工作线程中（作为 ``AudioInput`` 的监听函数），因此：
    - 只做推理和事件投递（``bus.publish_threadsafe``），不做任何业务判断；
    - 由事件循环线程通过 :meth:`set_vad_enabled` 等方法控制开关（简单的属性赋值，
      在 CPython 中是原子的，无需加锁）。

何时运行 VAD？
    - 唤醒后的聆听阶段（LISTENING）、追问窗口（FOLLOW_UP）：需要判断用户何时说完；
    - 播报 / 思考阶段且开启了 VAD 打断：需要判断用户是否开口；
    - 其余时间关闭，节省 CPU 并避免把环境噪音当成语音。
唤醒词检测则始终运行（打断依赖它）。
"""

from __future__ import annotations

import logging

import numpy as np

from pinecone.core.bus import EventBus
from pinecone.core.events import SpeechEnded, SpeechStarted, WakeWordDetected
from pinecone.speech.vad.base import SpeechEnd, SpeechStart, VoiceActivityDetector
from pinecone.speech.wake.base import WakeWordDetector

logger = logging.getLogger(__name__)


class SpeechPipeline:
    """唤醒词 + VAD 前端流水线。"""

    def __init__(self, bus: EventBus, wake: WakeWordDetector, vad: VoiceActivityDetector) -> None:
        self._bus = bus
        self._wake = wake
        self._vad = vad
        self._vad_enabled = False
        self._vad_reset_requested = False

    # ---- 事件循环线程调用 ----

    def set_vad_enabled(self, enabled: bool, *, reset: bool = True) -> None:
        """打开 / 关闭 VAD。``reset=True`` 时会在下一帧清空 VAD 状态（开始新一轮聆听）。"""
        if reset:
            self._vad_reset_requested = True
        self._vad_enabled = enabled

    def set_wake_threshold_boost(self, boost: float) -> None:
        """播报期间提高唤醒阈值。"""
        self._wake.set_threshold_boost(boost)

    # ---- 采集工作线程调用 ----

    def on_frame(self, frame: np.ndarray) -> None:
        """处理一帧音频（注册为 AudioInput 的监听函数）。"""
        keyword = self._wake.accept_frame(frame)
        if keyword:
            logger.info("检测到唤醒词：%s", keyword)
            # 唤醒词本身不应被当作用户指令，因此同时清空 VAD
            self._vad_reset_requested = True
            self._bus.publish_threadsafe(WakeWordDetected(keyword=keyword))
            return

        if self._vad_reset_requested:
            self._vad_reset_requested = False
            self._vad.reset()
        if not self._vad_enabled:
            return
        for event in self._vad.accept_frame(frame):
            if isinstance(event, SpeechStart):
                self._bus.publish_threadsafe(SpeechStarted())
            elif isinstance(event, SpeechEnd):
                self._bus.publish_threadsafe(SpeechEnded(audio=event.audio))
