"""本地离线语音合成：Matcha-TTS 中文模型 + vocos 声码器（通过 sherpa-onnx 调用）。

树莓派 5 上合成一句 10 字左右的话约 0.3~0.5 秒，可作为断网降级，
也可以把 ``tts.engine`` 设为 ``local`` 完全离线使用。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import numpy as np
import sherpa_onnx

from pinecone.core.config import TtsConfig
from pinecone.speech.tts.base import Audio, TextToSpeech

logger = logging.getLogger(__name__)


class LocalTTS(TextToSpeech):
    """sherpa-onnx Matcha 本地合成。

    线程安全：OfflineTts.generate 不保证可并发调用，这里用锁串行化。
    """

    name = "local"

    def __init__(self, config: TtsConfig, model_dir: Path) -> None:
        acoustic = model_dir / "model-steps-3.onnx"
        vocoder = model_dir / "vocos-22khz-univ.onnx"
        for path in (acoustic, vocoder):
            if not path.exists():
                raise FileNotFoundError(
                    f"找不到 {path}，请先运行 bash scripts/download_models.sh tts"
                )

        # rule_fsts：数字、日期、电话号码的读法规则，例如“3点20分”读作“三点二十分”
        rule_fsts = ",".join(
            str(model_dir / name)
            for name in ("phone.fst", "date.fst", "number.fst")
            if (model_dir / name).exists()
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=str(acoustic),
                    vocoder=str(vocoder),
                    lexicon=str(model_dir / "lexicon.txt"),
                    tokens=str(model_dir / "tokens.txt"),
                    dict_dir=str(model_dir / "dict"),
                ),
                num_threads=config.num_threads,
            ),
            rule_fsts=rule_fsts,
            max_num_sentences=1,
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._speed = config.local_speed
        self._lock = threading.Lock()
        logger.info("本地语音合成已加载：%s", model_dir)

    def _generate(self, text: str) -> Audio:
        with self._lock:
            audio = self._tts.generate(text, sid=0, speed=self._speed)
        return np.asarray(audio.samples, dtype=np.float32), audio.sample_rate

    async def synthesize(self, text: str) -> Audio:
        samples, rate = await asyncio.to_thread(self._generate, text)
        if samples.size == 0:
            raise RuntimeError(f"本地合成结果为空：{text}")
        return samples, rate
