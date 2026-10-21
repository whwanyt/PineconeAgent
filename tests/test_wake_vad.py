"""唤醒词与 VAD 的测试。

需要先下载模型（bash scripts/download_models.sh kws vad），模型不存在时自动跳过。
测试音频使用唤醒模型包自带的 test_wavs。
"""

import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from pinecone.core.bus import EventBus
from pinecone.core.config import PROJECT_ROOT, VadConfig, WakeConfig
from pinecone.core.events import SpeechEnded, WakeWordDetected
from pinecone.speech.pipeline import SpeechPipeline
from pinecone.speech.wake.sherpa_kws import keyword_to_tokens

KWS_DIR = PROJECT_ROOT / "models" / "kws"
VAD_MODEL = PROJECT_ROOT / "models" / "silero_vad.onnx"
needs_models = pytest.mark.skipif(
    not (KWS_DIR.exists() and VAD_MODEL.exists()), reason="未下载 kws / vad 模型"
)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def frames(audio: np.ndarray, size: int = 320):
    for start in range(0, audio.size - size + 1, size):
        yield audio[start : start + size]


def test_keyword_to_tokens() -> None:
    """中文唤醒词转换成声母 + 带调韵母，零声母字只保留韵母。"""
    assert keyword_to_tokens("小爱同学") == ["x", "iǎo", "ài", "t", "óng", "x", "ué"]


@needs_models
def test_sherpa_kws_detects_keyword() -> None:
    """自定义唤醒词能在测试音频中被检测到；提高阈值后不再触发。"""
    from pinecone.speech.wake.sherpa_kws import SherpaKeywordSpotter

    audio = np.concatenate([read_wav(KWS_DIR / "test_wavs" / "3.wav"), np.zeros(16000, np.float32)])
    kws = SherpaKeywordSpotter(WakeConfig(keywords=["法国"]), KWS_DIR)
    assert "法国" in [r for f in frames(audio) if (r := kws.accept_frame(f))]

    kws.set_threshold_boost(0.7)
    assert not [r for f in frames(audio) if (r := kws.accept_frame(f))]


@needs_models
async def test_pipeline_publishes_events() -> None:
    """流水线发布唤醒事件；开启 VAD 后能切出完整语音段。"""
    from pinecone.speech.vad.silero import SileroVad
    from pinecone.speech.wake.sherpa_kws import SherpaKeywordSpotter

    bus = EventBus()
    bus.bind_loop(asyncio.get_running_loop())
    got: list[object] = []
    bus.subscribe(WakeWordDetected, got.append)
    bus.subscribe(SpeechEnded, got.append)

    pipeline = SpeechPipeline(
        bus,
        SherpaKeywordSpotter(WakeConfig(keywords=["法国"]), KWS_DIR),
        SileroVad(VadConfig(), VAD_MODEL),
    )
    pipeline.set_vad_enabled(True)
    speech = read_wav(KWS_DIR / "test_wavs" / "5.wav")  # 不含唤醒词的语音
    for f in frames(np.concatenate([speech, np.zeros(16000, np.float32)])):
        pipeline.on_frame(f)
    await asyncio.sleep(0.05)
    ended = [e for e in got if isinstance(e, SpeechEnded)]
    assert ended and ended[0].audio.size > 16000 * 0.3
