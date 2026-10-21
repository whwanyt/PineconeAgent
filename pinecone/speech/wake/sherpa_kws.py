"""基于 sherpa-onnx 关键词检测（KWS）的唤醒词实现。

为什么选它？
    - 模型只有 3.3M 参数，树莓派 5 单线程实时率远小于 0.1；
    - **开放词表**：唤醒词直接写中文，程序自动转成拼音音素，不需要采集数据训练模型；
    - 可以同时配置多个唤醒词。

关键词格式（模型要求的“拼音音素”形式）::

    x iǎo p ài x iǎo p ài @小派小派
    └──────── 音素 ───────┘ └显示文本┘

声母和带声调的韵母分开，零声母字（如“爱”）只有韵母。

关于“播报期间提高阈值”：
    实测 sherpa-onnx 1.13 中关键词行内的 ``#阈值`` 写法在 ``create_stream`` 时不生效，
    只有构造函数的全局 ``keywords_threshold`` 有效。因此这里按阈值缓存多个
    ``KeywordSpotter`` 实例（模型很小，多一个实例只多占几 MB 内存），切换阈值时换用对应实例。
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
import sherpa_onnx
from pypinyin import Style, pinyin

from pinecone.core.config import WakeConfig
from pinecone.speech.wake.base import WakeWordDetector

logger = logging.getLogger(__name__)

# 阈值上限：再高基本无法唤醒
_MAX_THRESHOLD = 0.95


def keyword_to_tokens(text: str) -> list[str]:
    """把中文唤醒词转换成模型使用的拼音音素序列。

    例如 ``"小爱同学"`` → ``["x", "iǎo", "ài", "t", "óng", "x", "ué"]``。

    ``strict=False`` 会把 y / w 也当作声母（与模型训练时的切分方式一致）。
    """
    initials = pinyin(text, style=Style.INITIALS, strict=False)
    finals = pinyin(text, style=Style.FINALS_TONE, strict=False)
    tokens: list[str] = []
    for (initial,), (final,) in zip(initials, finals, strict=True):
        if initial:
            tokens.append(initial)
        if final:
            tokens.append(final)
    return tokens


def _find_model_file(model_dir: Path, prefix: str, use_int8: bool) -> Path:
    """在模型目录中查找 encoder/decoder/joiner 文件。

    官方模型包里同时有 epoch-12 与 epoch-99 两套、以及 int8 / fp32 两种精度，
    这里按“精度匹配优先、文件名排序后取第一个”的规则选择，避免把文件名写死在配置里。
    """
    candidates = sorted(model_dir.glob(f"{prefix}-*.onnx"))
    matched = [p for p in candidates if (".int8." in p.name) == use_int8]
    chosen = (matched or candidates)[:1]
    if not chosen:
        raise FileNotFoundError(
            f"在 {model_dir} 中找不到 {prefix}*.onnx，请先运行 bash scripts/download_models.sh kws"
        )
    return chosen[0]


class SherpaKeywordSpotter(WakeWordDetector):
    """sherpa-onnx 唤醒词检测器。"""

    def __init__(self, config: WakeConfig, model_dir: Path, sample_rate: int = 16000) -> None:
        self._config = config
        self._rate = sample_rate
        self._model_dir = model_dir
        tokens_path = model_dir / "tokens.txt"
        if not tokens_path.exists():
            raise FileNotFoundError(
                f"找不到唤醒模型 {model_dir}，请先运行 bash scripts/download_models.sh kws"
            )
        self._vocab = set(tokens_path.read_text(encoding="utf-8").split())
        # 校验唤醒词：每个音素都必须在模型词表中，否则模型永远检测不到
        keyword_lines = [f"{' '.join(self._validate(kw))} @{kw}" for kw in config.keywords]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("\n".join(keyword_lines))
            self._keywords_file = Path(f.name)

        # 按阈值缓存的检测器实例；_active 为当前使用的（阈值, 检测器, 流）
        self._spotters: dict[float, sherpa_onnx.KeywordSpotter] = {}
        self._lock = threading.Lock()
        self._pending_boost: float | None = None
        self._boost = 0.0
        self._spotter = self._get_spotter(config.threshold)
        self._stream = self._spotter.create_stream()
        logger.info("唤醒词已加载：%s（阈值 %.2f）", "、".join(config.keywords), config.threshold)

    def _validate(self, keyword: str) -> list[str]:
        tokens = keyword_to_tokens(keyword)
        missing = [t for t in tokens if t not in self._vocab]
        if missing:
            raise ValueError(f"唤醒词「{keyword}」包含模型不支持的音素：{missing}")
        return tokens

    def _get_spotter(self, threshold: float) -> sherpa_onnx.KeywordSpotter:
        """获取（必要时创建）指定阈值的检测器实例。"""
        threshold = round(min(_MAX_THRESHOLD, threshold), 2)
        if threshold not in self._spotters:
            cfg = self._config
            self._spotters[threshold] = sherpa_onnx.KeywordSpotter(
                tokens=str(self._model_dir / "tokens.txt"),
                encoder=str(_find_model_file(self._model_dir, "encoder", cfg.use_int8)),
                decoder=str(_find_model_file(self._model_dir, "decoder", cfg.use_int8)),
                joiner=str(_find_model_file(self._model_dir, "joiner", cfg.use_int8)),
                keywords_file=str(self._keywords_file),
                num_threads=cfg.num_threads,
                sample_rate=self._rate,
                keywords_score=cfg.score,
                keywords_threshold=threshold,
                # 唤醒词后面至少跟 1 帧空白才触发，减少“小派小派……”说到一半就触发的情况
                num_trailing_blanks=1,
            )
        return self._spotters[threshold]

    # ------------------------------------------------------------------ 接口实现

    def set_threshold_boost(self, boost: float) -> None:
        """线程安全：只记录待生效的值，真正切换在工作线程的下一帧进行。"""
        with self._lock:
            self._pending_boost = boost

    def reset(self) -> None:
        self._spotter.reset_stream(self._stream)

    def accept_frame(self, frame: np.ndarray) -> str | None:
        with self._lock:
            pending, self._pending_boost = self._pending_boost, None
        if pending is not None and pending != self._boost:
            # 切换到另一个阈值的检测器并新建流。会丢失约几百毫秒的上下文，
            # 只发生在开始 / 结束播报的瞬间，影响可以忽略。
            self._boost = pending
            self._spotter = self._get_spotter(self._config.threshold + pending)
            self._stream = self._spotter.create_stream()

        self._stream.accept_waveform(self._rate, frame)
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                # 触发后必须重置，否则同一段音频会重复触发
                self._spotter.reset_stream(self._stream)
                return result
        return None
