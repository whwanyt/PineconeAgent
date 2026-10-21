"""基于 Silero VAD（通过 sherpa-onnx 调用）的语音活动检测。

Silero VAD 是目前开源 VAD 中准确率与速度兼顾最好的之一，
模型不到 2MB，树莓派上每帧推理不到 1ms。

sherpa-onnx 的 ``VoiceActivityDetector`` 内部会：
    - 把任意长度的输入攒成 512 样本的窗口逐个推理；
    - 按 min_silence / min_speech 规则切分出完整语音段，放进内部队列；
我们只需要：检测“是否刚开始说话”（发 SpeechStart），以及从队列取出完整语音段（发 SpeechEnd）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

from pinecone.core.config import VadConfig
from pinecone.speech.vad.base import SpeechEnd, SpeechStart, VadEvent, VoiceActivityDetector

# 内部缓冲最长保留的音频时长（秒），需要大于 max_speech_s
_BUFFER_SECONDS = 60


class SileroVad(VoiceActivityDetector):
    """Silero VAD 实现。"""

    def __init__(self, config: VadConfig, model_path: Path, sample_rate: int = 16000) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到 VAD 模型 {model_path}，请先运行 bash scripts/download_models.sh vad"
            )
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = str(model_path)
        vad_config.silero_vad.threshold = config.threshold
        vad_config.silero_vad.min_silence_duration = config.min_silence_ms / 1000
        vad_config.silero_vad.min_speech_duration = config.min_speech_ms / 1000
        vad_config.silero_vad.max_speech_duration = config.max_speech_s
        vad_config.sample_rate = sample_rate
        vad_config.num_threads = 1
        self._vad = sherpa_onnx.VoiceActivityDetector(
            vad_config, buffer_size_in_seconds=_BUFFER_SECONDS
        )
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._vad.reset()
        self._in_speech = False

    def accept_frame(self, frame: np.ndarray) -> list[VadEvent]:
        events: list[VadEvent] = []
        self._vad.accept_waveform(frame)

        detected = self._vad.is_speech_detected()
        if detected and not self._in_speech:
            events.append(SpeechStart())
        self._in_speech = detected

        # 一个完整语音段结束后才会进入队列
        while not self._vad.empty():
            samples = np.array(self._vad.front.samples, dtype=np.float32)
            self._vad.pop()
            events.append(SpeechEnd(audio=samples))
            self._in_speech = False
        return events
