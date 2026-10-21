"""微软 Edge 在线语音合成（edge-tts）。

优点：免费、无需 API Key、中文音色非常自然。
缺点：依赖网络，首包延迟约 0.5~1.5 秒；因此配合本地引擎做降级。

返回的是 24kHz mp3，用 miniaudio 解码成 PCM。
"""

from __future__ import annotations

import asyncio

import edge_tts
import miniaudio
import numpy as np

from pinecone.audio.utils import int16_to_float
from pinecone.core.config import TtsConfig
from pinecone.speech.tts.base import Audio, TextToSpeech

_SAMPLE_RATE = 24000


class EdgeTTS(TextToSpeech):
    """edge-tts 引擎。"""

    name = "edge"

    def __init__(self, config: TtsConfig) -> None:
        self._config = config

    async def synthesize(self, text: str) -> Audio:
        """合成一句话。超过 ``edge_timeout_s`` 抛出 TimeoutError（交给降级逻辑处理）。"""
        async with asyncio.timeout(self._config.edge_timeout_s):
            mp3 = bytearray()
            communicate = edge_tts.Communicate(
                text, self._config.edge_voice, rate=self._config.edge_rate
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3.extend(chunk["data"])
        if not mp3:
            raise RuntimeError("edge-tts 没有返回音频")
        decoded = miniaudio.decode(
            bytes(mp3),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=_SAMPLE_RATE,
        )
        samples = int16_to_float(np.asarray(decoded.samples, dtype=np.int16))
        return samples, decoded.sample_rate
