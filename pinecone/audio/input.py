"""麦克风采集。

线程模型（重要）::

    PortAudio 回调线程            采集工作线程                         事件循环线程
    ───────────────────          ─────────────────────────           ─────────────
    _callback()                   _worker()
      复制数据 → 队列 ──────────▶  重采样到 16kHz
      （必须极快，不能做计算）       切成固定长度帧
                                   回声消除 processor(frame)
                                   依次调用 listener(frame)  ──事件──▶ bus.publish_threadsafe
                                   （唤醒词、VAD 的推理在这里）

为什么不在回调里直接做推理？
    PortAudio 回调有严格的实时性要求，耗时操作会导致丢帧（input overflow）。
    唤醒词 / VAD 推理每帧需要几毫秒，放到独立工作线程里最稳妥。

为什么会有重采样？
    很多 USB 麦克风只支持 44.1k/48k，打开 16k 会失败，因此先尝试 16k，不行就用设备默认采样率
    采集，再在工作线程中重采样。
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from pinecone.audio.devices import device_info, resolve_device
from pinecone.audio.utils import resample
from pinecone.core.config import AudioConfig

logger = logging.getLogger(__name__)

# 监听函数：在采集工作线程中被调用，参数为 16kHz float32 单声道帧
FrameListener = Callable[[np.ndarray], None]
# 帧处理器：用于回声消除，输入原始帧，输出处理后的帧（长度不变）
FrameProcessor = Callable[[np.ndarray], np.ndarray]

# 回调线程与工作线程之间的队列长度（块数）。工作线程卡顿超过约 2 秒才会开始丢数据。
_QUEUE_BLOCKS = 100


class AudioInput:
    """麦克风采集器。

    用法::

        mic = AudioInput(cfg.audio)
        mic.add_listener(kws.accept_frame)
        mic.start()
        ...
        mic.stop()

    线程安全：``add_listener`` / ``set_processor`` 可在任意线程调用；
    listener 和 processor 都运行在采集工作线程中。
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self.sample_rate = config.sample_rate
        self.frame_size = config.sample_rate * config.frame_ms // 1000
        self._listeners: list[FrameListener] = []
        self._processor: FrameProcessor | None = None
        self._lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=_QUEUE_BLOCKS)
        self._stream: sd.InputStream | None = None
        self._worker: threading.Thread | None = None
        self._device_rate = config.sample_rate
        self._overflows = 0

    # ------------------------------------------------------------------ 配置

    def add_listener(self, listener: FrameListener) -> None:
        """注册帧监听函数（在工作线程中调用，必须在一帧时长内处理完，否则会积压）。"""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: FrameListener) -> None:
        """移除帧监听函数。"""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def set_processor(self, processor: FrameProcessor | None) -> None:
        """设置帧处理器（回声消除），``None`` 表示不处理。"""
        with self._lock:
            self._processor = processor

    # ------------------------------------------------------------------ 启停

    def start(self) -> None:
        """打开麦克风并启动工作线程。"""
        device = resolve_device(self._config.input_device, "input")
        self._device_rate = self._pick_sample_rate(device)
        block = self._device_rate * self._config.frame_ms // 1000

        self._worker = threading.Thread(target=self._work, name="audio-input", daemon=True)
        self._worker.start()
        self._stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=self._device_rate,
            blocksize=block,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "麦克风已启动：%s，设备采样率 %dHz → 模型采样率 %dHz",
            device_info(device, "input")["name"],
            self._device_rate,
            self.sample_rate,
        )

    def stop(self) -> None:
        """关闭麦克风并停止工作线程。可重复调用。"""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._queue.put(None)  # 哨兵：通知工作线程退出
            self._worker.join(timeout=2)
            self._worker = None

    def _pick_sample_rate(self, device: int | None) -> int:
        """优先使用 16kHz 采集；设备不支持时退回设备默认采样率。"""
        try:
            sd.check_input_settings(device=device, samplerate=self.sample_rate, channels=1)
            return self.sample_rate
        except Exception:
            rate = int(device_info(device, "input")["default_samplerate"])
            logger.info("麦克风不支持 %dHz，改用 %dHz 采集后重采样", self.sample_rate, rate)
            return rate

    # ------------------------------------------------------------------ 线程

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        """PortAudio 回调（实时线程）：只做复制和入队，绝不做任何计算或加锁等待。"""
        if status.input_overflow:
            self._overflows += 1
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # 工作线程跟不上时丢弃最新数据，并计数，便于排查树莓派性能问题
            self._overflows += 1

    def _work(self) -> None:
        """采集工作线程：重采样 → 切帧 → 回声消除 → 分发给监听函数。"""
        pending = np.zeros(0, dtype=np.float32)
        while True:
            block = self._queue.get()
            if block is None:
                return
            if self._device_rate != self.sample_rate:
                block = resample(block, self._device_rate, self.sample_rate)
            # 重采样后的长度不一定正好是一帧，攒够整帧再处理，保证下游模型拿到的帧长固定
            pending = np.concatenate([pending, block])
            while pending.size >= self.frame_size:
                frame, pending = pending[: self.frame_size], pending[self.frame_size :]
                self._dispatch(frame)

    def _dispatch(self, frame: np.ndarray) -> None:
        """对一帧执行回声消除并分发；单个监听函数出错不影响其他监听函数。"""
        with self._lock:
            processor = self._processor
            listeners = list(self._listeners)
        if processor is not None:
            try:
                frame = processor(frame)
            except Exception:
                logger.exception("回声消除处理失败，本帧使用原始音频")
        for listener in listeners:
            try:
                listener(frame)
            except Exception:
                logger.exception(
                    "音频帧监听函数出错：%s", getattr(listener, "__qualname__", listener)
                )
