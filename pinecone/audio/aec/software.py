"""进程内软件回声消除（aec.mode = software）。

为什么不需要额外硬件也能做？
    回声消除需要两路信号：麦克风信号（近端，包含用户说话 + 扬声器回声）和
    扬声器正在播放的信号（远端 / 参考信号）。我们的语音播报全部经过 ``AudioOutput``
    的同一个输出流，它在每次回调时都会把输出样本交给本模块，因此天然拥有参考信号。

数据流::

    AudioOutput 回调线程（48kHz）            AudioInput 工作线程（16kHz）
    ─────────────────────────────           ─────────────────────────────
    _on_reference(block, 48000)              process(mic_frame)
      重采样到 16kHz                            从参考缓冲取等长样本 ref
      追加到参考缓冲（加锁，极短）──────────▶    后端.process(mic_frame, ref)
                                                返回去回声后的帧

延迟对齐：
    参考信号在“交给声卡”时就进入缓冲，而它真正从扬声器发出、再被麦克风录到，
    要经过输出缓冲 + 声学传播 + 输入缓冲，大约几十毫秒。
    - WebRTC 后端（AEC3）自带延迟估计，``stream_delay_ms`` 只是初始提示；
    - speex 后端靠足够长的自适应滤波器（``filter_ms``）覆盖这段延迟。
    参考缓冲最多保留 1 秒，防止两路时钟漂移导致无限堆积。

局限：
    音乐由 mpv 独立播放，不经过 AudioOutput，因此 **音乐的回声不在参考信号里**。
    播放音乐时请依靠“唤醒后自动压低音乐音量”，或使用 pipewire 模式做系统级消除。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Protocol

import numpy as np

from pinecone.audio.aec.base import EchoCanceller
from pinecone.audio.utils import float_to_int16, int16_to_float, resample

if TYPE_CHECKING:
    from pinecone.audio.input import AudioInput
    from pinecone.audio.output import AudioOutput
    from pinecone.core.config import AudioConfig

logger = logging.getLogger(__name__)

# 参考缓冲最多保留的时长（秒）
_MAX_REF_SECONDS = 1.0


class _Backend(Protocol):
    """算法后端接口：输入等长的近端帧和参考帧（16kHz float32），返回去回声后的帧。"""

    name: str

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray: ...


class _WebRtcBackend:
    """WebRTC AudioProcessingModule（来自 pip 包 ``livekit``），效果最好。

    APM 要求每次输入正好 10ms 的音频，因此 20ms 的帧会被拆成两段处理。
    参考信号必须先于对应的麦克风信号送入（process_reverse_stream 再 process_stream）。
    """

    name = "webrtc"

    def __init__(self, sample_rate: int, delay_ms: int, noise_suppression: bool) -> None:
        from livekit import rtc  # 延迟导入：未安装时由工厂函数捕获 ImportError

        self._rtc = rtc
        self._rate = sample_rate
        self._chunk = sample_rate // 100  # 10ms
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=True,
            auto_gain_control=False,  # AGC 会干扰唤醒词模型的阈值判断，关闭
        )
        self._apm.set_stream_delay_ms(delay_ms)

    def _frame(self, samples: np.ndarray):  # noqa: ANN202 - livekit 类型
        pcm = float_to_int16(samples)
        return self._rtc.AudioFrame(
            data=pcm.tobytes(), sample_rate=self._rate, num_channels=1, samples_per_channel=pcm.size
        )

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        out = np.empty_like(near)
        for start in range(0, near.size - self._chunk + 1, self._chunk):
            end = start + self._chunk
            self._apm.process_reverse_stream(self._frame(far[start:end]))
            frame = self._frame(near[start:end])
            self._apm.process_stream(frame)  # 原地修改 frame.data
            out[start:end] = int16_to_float(np.frombuffer(frame.data, dtype=np.int16))
        return out


class _SpeexBackend:
    """speexdsp 回声消除（pip 包 ``speexdsp``，需要系统库 libspeexdsp-dev），轻量的降级方案。"""

    name = "speex"

    def __init__(self, sample_rate: int, frame_size: int, filter_ms: int) -> None:
        from speexdsp import EchoCanceller as SpeexEC

        filter_length = sample_rate * filter_ms // 1000
        self._ec = SpeexEC.create(frame_size, filter_length, sample_rate)

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        out = self._ec.process(float_to_int16(near).tobytes(), float_to_int16(far).tobytes())
        return int16_to_float(out)


class SoftwareEchoCanceller(EchoCanceller):
    """进程内软件回声消除。没有可用后端时自动降级为不处理（``effective=False``）。"""

    mode = "software"

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._rate = config.sample_rate
        self._frame_size = config.sample_rate * config.frame_ms // 1000
        self._ref: deque[float] = deque(maxlen=int(self._rate * _MAX_REF_SECONDS))
        self._lock = threading.Lock()
        self._backend: _Backend | None = self._create_backend()

    def _create_backend(self) -> _Backend | None:
        """按配置创建后端；auto 模式依次尝试 webrtc、speex。"""
        aec = self._config.aec
        candidates = ["webrtc", "speex"] if aec.backend == "auto" else [aec.backend]
        for name in candidates:
            try:
                if name == "webrtc":
                    backend: _Backend = _WebRtcBackend(
                        self._rate, aec.stream_delay_ms, aec.noise_suppression
                    )
                else:
                    backend = _SpeexBackend(self._rate, self._frame_size, aec.filter_ms)
                logger.info("软件回声消除已启用，后端：%s", name)
                return backend
            except ImportError:
                logger.debug("回声消除后端 %s 不可用（未安装）", name)
            except Exception:
                logger.exception("回声消除后端 %s 初始化失败", name)
        logger.warning(
            "没有可用的软件回声消除后端，已降级为不处理。"
            "请安装：uv pip install 'pinecone-agent[aec]'（webrtc）或 '[aec-speex]'"
        )
        return None

    @property
    def effective(self) -> bool:
        return self._backend is not None

    def attach(self, mic: AudioInput, speaker: AudioOutput) -> None:
        if self._backend is None:
            mic.set_processor(None)
            return
        speaker.add_reference_listener(self._on_reference)
        mic.set_processor(self.process)

    # ---- 参考信号（AudioOutput 回调线程）----

    def _on_reference(self, block: np.ndarray, sample_rate: int) -> None:
        """接收扬声器输出块：重采样到模型采样率后追加到参考缓冲。"""
        ref = resample(block, sample_rate, self._rate)
        with self._lock:
            self._ref.extend(ref.tolist())

    # ---- 麦克风帧（AudioInput 工作线程）----

    def _take_reference(self, n: int) -> np.ndarray:
        """取出 n 个参考样本；不足时用 0 补齐（扬声器静音时就是这种情况）。"""
        with self._lock:
            k = min(n, len(self._ref))
            taken = [self._ref.popleft() for _ in range(k)]
        far = np.zeros(n, dtype=np.float32)
        far[:k] = taken
        return far

    def process(self, frame: np.ndarray) -> np.ndarray:
        """对一帧麦克风信号做回声消除。"""
        if self._backend is None:
            return frame
        far = self._take_reference(frame.size)
        return self._backend.process(frame, far)
