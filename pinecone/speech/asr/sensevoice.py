"""SenseVoice-Small 本地语音识别（通过 sherpa-onnx 调用）。

SenseVoice 是阿里开源的多语种识别模型（中/英/日/韩/粤），非自回归结构，
速度比 Whisper 快一个数量级，中文准确率也更高，非常适合树莓派这类设备。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import sherpa_onnx

from pinecone.core.config import AsrConfig
from pinecone.speech.asr.base import SpeechRecognizer

logger = logging.getLogger(__name__)

# SenseVoice 偶尔会输出 <|zh|><|NEUTRAL|> 这类情感 / 事件标签，sherpa-onnx 一般已去除，这里兜底
_TAG_PATTERN = re.compile(r"<\|[^|]*\|>")


class SenseVoiceRecognizer(SpeechRecognizer):
    """SenseVoice 识别器。

    线程安全：sherpa-onnx 的 OfflineRecognizer 支持多线程各自创建 stream 并行解码，
    但我们一次只识别一句话，不存在并发。
    """

    def __init__(self, config: AsrConfig, model_dir: Path) -> None:
        model = self._pick_model(model_dir)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=config.num_threads,
            sample_rate=self.sample_rate,
            language="" if config.language == "auto" else config.language,
            use_itn=config.use_itn,
        )
        logger.info("语音识别模型已加载：%s", model.name)

    @staticmethod
    def _pick_model(model_dir: Path) -> Path:
        """优先使用 int8 量化模型（体积小 4 倍，速度更快，精度几乎无损）。"""
        for name in ("model.int8.onnx", "model.onnx"):
            if (model_dir / name).exists():
                return model_dir / name
        raise FileNotFoundError(
            f"在 {model_dir} 中找不到 SenseVoice 模型，请先运行 bash scripts/download_models.sh asr"
        )

    def recognize(self, audio: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio.astype(np.float32, copy=False))
        self._recognizer.decode_stream(stream)
        return _TAG_PATTERN.sub("", stream.result.text).strip()
