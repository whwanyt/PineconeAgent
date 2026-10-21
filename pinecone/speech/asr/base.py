"""语音识别接口与工厂函数。

识别是“整句识别”：VAD 切出一句完整的话后一次性识别。
相比流式识别，整句识别准确率更高，且 SenseVoice 速度足够快（树莓派 5 上 5 秒语音约 0.3 秒），
对用户来说延迟几乎无感。

扩展云端识别：继承 :class:`SpeechRecognizer`，实现 :meth:`recognize`（同步阻塞版本即可，
会被放到线程池执行），或者直接重写 :meth:`transcribe` 写成异步 HTTP 调用。
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pinecone.core.config import AppConfig

logger = logging.getLogger(__name__)


class SpeechRecognizer(ABC):
    """语音识别器。"""

    sample_rate: int = 16000

    @abstractmethod
    def recognize(self, audio: np.ndarray) -> str:
        """同步识别一段 16kHz float32 音频，返回文本（可能为空字符串）。

        会阻塞调用线程，请通过 :meth:`transcribe` 在线程池中调用。
        """

    async def transcribe(self, audio: np.ndarray) -> str:
        """异步识别：在线程池中执行 :meth:`recognize`，不阻塞事件循环。

        可取消：调用方被取消时立即返回（后台线程会把这次识别跑完，但结果被丢弃）。
        """
        start = time.monotonic()
        text = (await asyncio.to_thread(self.recognize, audio)).strip()
        logger.info(
            "识别结果（%.1fs 语音，耗时 %.2fs）：%s",
            audio.size / self.sample_rate,
            time.monotonic() - start,
            text or "<空>",
        )
        return text


def create_recognizer(config: AppConfig) -> SpeechRecognizer:
    """根据配置创建识别器。"""
    if config.asr.engine == "sensevoice":
        from pinecone.speech.asr.sensevoice import SenseVoiceRecognizer

        return SenseVoiceRecognizer(config.asr, config.resolve_path(config.asr.model_dir))
    raise ValueError(f"不支持的识别引擎：{config.asr.engine}")
