"""语音合成接口、降级组合与工厂函数。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pinecone.core.config import AppConfig

logger = logging.getLogger(__name__)

# 合成结果：(float32 单声道样本, 采样率)
Audio = tuple[np.ndarray, int]


class TextToSpeech(ABC):
    """语音合成引擎。"""

    name: str = ""

    @abstractmethod
    async def synthesize(self, text: str) -> Audio:
        """把一句文本合成为音频。

        可取消：调用方被取消时应尽快返回（网络请求会被中断，本地合成在后台线程跑完后丢弃）。

        Raises:
            Exception: 合成失败（网络错误、超时等），由降级逻辑捕获。
        """


class FallbackTTS(TextToSpeech):
    """带降级和缓存的组合引擎：先用主引擎，失败时用降级引擎。

    缓存：短句（不超过 20 字）的合成结果按文本缓存（LRU），像“好的”“已停止”这类高频回复
    第二次播报时没有任何合成延迟，断网时也能使用缓存。
    """

    name = "fallback"
    _CACHE_MAX_CHARS = 20

    def __init__(
        self, primary: TextToSpeech, fallback: TextToSpeech | None, cache_size: int
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._cache: OrderedDict[str, Audio] = OrderedDict()
        self._cache_size = cache_size

    async def synthesize(self, text: str) -> Audio:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        try:
            audio = await self._primary.synthesize(text)
        except Exception as exc:
            if self._fallback is None:
                raise
            logger.warning(
                "%s 合成失败（%s），改用 %s", self._primary.name, exc, self._fallback.name
            )
            audio = await self._fallback.synthesize(text)
        if self._cache_size > 0 and len(text) <= self._CACHE_MAX_CHARS:
            self._cache[text] = audio
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return audio


def create_tts(config: AppConfig) -> TextToSpeech:
    """根据配置创建（带降级的）合成引擎。

    本地模型不存在时不会报错，只是没有降级能力（并打印警告）。
    """
    from pinecone.speech.tts.edge import EdgeTTS
    from pinecone.speech.tts.local import LocalTTS

    tts_cfg = config.tts
    local: TextToSpeech | None = None
    if tts_cfg.engine == "local" or tts_cfg.fallback == "local":
        try:
            local = LocalTTS(tts_cfg, config.resolve_path(tts_cfg.local_model_dir))
        except FileNotFoundError as exc:
            if tts_cfg.engine == "local":
                raise
            logger.warning("本地语音合成不可用，断网时将无法播报：%s", exc)

    if tts_cfg.engine == "local":
        assert local is not None
        return FallbackTTS(local, None, tts_cfg.cache_size)
    return FallbackTTS(EdgeTTS(tts_cfg), local, tts_cfg.cache_size)
