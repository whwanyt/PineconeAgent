"""扬声器播放（可取消、可流式追加、可提供回声消除参考信号）。

设计要点：

1. **单一常开输出流 + 软件混音**：
   整个程序只打开一个输出流，所有语音播报都排队写入。这样我们能精确知道“此刻扬声器
   正在放什么”，并把它作为参考信号交给软件回声消除（见 ``audio/aec/software.py``）。

2. **流式播放**：
   TTS 是边合成边播放的，:class:`PlaybackHandle` 支持不断 ``feed`` 新的音频块，
   ``finish`` 后等缓冲区播完才算结束。

3. **可取消**：
   ``await output.play(...)`` 所在任务被取消时（例如被唤醒词打断），会立即停止这段播放，
   扬声器在下一个音频块（约 20ms）内静音。

线程模型：
    - ``_callback`` 运行在 PortAudio 实时线程，只做数组拷贝和简单乘法；
    - 播放完成的通知通过 ``loop.call_soon_threadsafe`` 投递回事件循环。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from pinecone.audio.devices import device_info, resolve_device
from pinecone.audio.utils import resample
from pinecone.core.config import AudioConfig

logger = logging.getLogger(__name__)

# 参考信号监听函数：在 PortAudio 回调线程中被调用，参数为（本块输出的单声道样本，输出采样率）。
# 必须极快返回（通常只是写入环形缓冲）。
ReferenceListener = Callable[[np.ndarray, int], None]


class PlaybackHandle:
    """一段（可能是流式的）播放。

    生命周期：创建 → 多次 :meth:`feed` → :meth:`finish` → 播放完毕 / :meth:`stop` → :meth:`wait` 返回。

    线程安全：``feed`` / ``finish`` / ``stop`` 可以在任意线程调用。
    """

    def __init__(self, output_rate: int, loop: asyncio.AbstractEventLoop | None) -> None:
        self._output_rate = output_rate
        self._loop = loop
        self._chunks: deque[np.ndarray] = deque()
        self._offset = 0  # 队首块已经读取的样本数
        self._lock = threading.Lock()
        self._finished = False
        self._stopped = False
        self._done = threading.Event()
        self._future: asyncio.Future[None] | None = loop.create_future() if loop else None

    # ---- 生产者侧 ----

    def feed(self, samples: np.ndarray, sample_rate: int) -> None:
        """追加一块音频（自动重采样到输出采样率）。已停止的句柄会忽略新数据。"""
        if samples.size == 0:
            return
        data = resample(samples, sample_rate, self._output_rate)
        with self._lock:
            if not self._stopped:
                self._chunks.append(data)

    def finish(self) -> None:
        """声明不会再有新数据，缓冲区播完后本段播放结束。"""
        with self._lock:
            self._finished = True

    def stop(self) -> None:
        """立即停止并丢弃尚未播放的数据。"""
        with self._lock:
            self._stopped = True
            self._chunks.clear()
        self._mark_done()

    @property
    def done(self) -> bool:
        """是否已结束（播完或被停止）。"""
        return self._done.is_set()

    async def wait(self) -> None:
        """等待播放结束。

        可取消：被取消时会调用 :meth:`stop` 立即静音，然后重新抛出 ``CancelledError``。
        """
        if self._future is None:
            await asyncio.to_thread(self._done.wait)
            return
        try:
            await asyncio.shield(self._future)
        except asyncio.CancelledError:
            self.stop()
            raise

    # ---- 消费者侧（PortAudio 回调线程）----

    def read_into(self, out: np.ndarray) -> int:
        """把尽可能多的数据写入 ``out``，返回写入的样本数；数据耗尽且已 finish 时标记完成。"""
        written = 0
        with self._lock:
            while written < out.size and self._chunks:
                chunk = self._chunks[0]
                n = min(out.size - written, chunk.size - self._offset)
                out[written : written + n] = chunk[self._offset : self._offset + n]
                written += n
                self._offset += n
                if self._offset >= chunk.size:
                    self._chunks.popleft()
                    self._offset = 0
            drained = not self._chunks and (self._finished or self._stopped)
        if drained:
            self._mark_done()
        return written

    def _mark_done(self) -> None:
        if self._done.is_set():
            return
        self._done.set()
        if self._future is not None and self._loop is not None and not self._loop.is_closed():
            future = self._future
            self._loop.call_soon_threadsafe(lambda: future.done() or future.set_result(None))


class AudioOutput:
    """语音播报输出器（单一输出流，播放句柄排队依次播放）。"""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self.volume = config.volume
        self.sample_rate = 48000
        self._channels = 1
        self._queue: deque[PlaybackHandle] = deque()
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._ref_listeners: list[ReferenceListener] = []

    # ------------------------------------------------------------------ 启停

    def start(self) -> None:
        """打开扬声器输出流（常开，没有数据时输出静音）。"""
        device = resolve_device(self._config.output_device, "output")
        info = device_info(device, "output")
        self.sample_rate = int(info["default_samplerate"])
        # 多数设备是立体声；单声道数据在回调中复制到所有声道
        self._channels = max(1, min(2, int(info["max_output_channels"])))
        self._stream = sd.OutputStream(
            device=device,
            channels=self._channels,
            samplerate=self.sample_rate,
            blocksize=self.sample_rate * self._config.frame_ms // 1000,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "扬声器已启动：%s，%dHz，%d 声道", info["name"], self.sample_rate, self._channels
        )

    def stop(self) -> None:
        """关闭输出流。"""
        self.stop_all()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------ 播放

    def open(self) -> PlaybackHandle:
        """创建一个流式播放句柄并排入播放队列（排在已有播放之后）。

        必须在事件循环线程中调用（需要获取当前事件循环来创建完成通知）。
        """
        handle = PlaybackHandle(self.sample_rate, asyncio.get_running_loop())
        with self._lock:
            self._queue.append(handle)
        return handle

    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """播放一段完整音频并等待播完。

        可取消：任务被取消时立即停止本段播放。
        """
        handle = self.open()
        handle.feed(samples, sample_rate)
        handle.finish()
        await handle.wait()

    def stop_all(self) -> None:
        """立即停止所有正在播放和排队的音频（“别说了”/打断时使用）。"""
        with self._lock:
            handles = list(self._queue)
            self._queue.clear()
        for handle in handles:
            handle.stop()

    @property
    def is_playing(self) -> bool:
        """当前是否有语音在播放或排队。"""
        with self._lock:
            return any(not h.done for h in self._queue)

    def add_reference_listener(self, listener: ReferenceListener) -> None:
        """注册参考信号监听函数（软件回声消除使用）。"""
        self._ref_listeners.append(listener)

    # ------------------------------------------------------------------ 混音

    def mix(self, frames: int) -> np.ndarray:
        """生成接下来 ``frames`` 个输出样本（单声道）。

        依次从队首句柄读取数据：当前句柄读完且已结束就切到下一个；
        当前句柄暂时没数据（TTS 还在合成）就输出静音等待，不跳到下一个，保证顺序。
        独立成方法是为了脱离声卡做单元测试。
        """
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < frames and self._queue:
                handle = self._queue[0]
                filled += handle.read_into(out[filled:])
                if handle.done:
                    self._queue.popleft()
                elif filled < frames:
                    break  # 当前句柄数据不足（欠载），剩余部分保持静音
        out *= self.volume
        return out

    def _callback(
        self, outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        """PortAudio 回调（实时线程）。"""
        mono = self.mix(frames)
        outdata[:] = mono[:, None]
        for listener in self._ref_listeners:
            try:
                listener(mono, self.sample_rate)
            except Exception:
                logger.exception("参考信号监听函数出错")
