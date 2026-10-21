"""SenseVoice 识别测试（需要先下载 asr 模型，否则跳过）。"""

import wave

import numpy as np
import pytest

from pinecone.core.config import PROJECT_ROOT, AsrConfig

ASR_DIR = PROJECT_ROOT / "models" / "asr"
needs_model = pytest.mark.skipif(not (ASR_DIR / "tokens.txt").exists(), reason="未下载 asr 模型")


@needs_model
async def test_sensevoice_recognizes_chinese() -> None:
    """模型包自带的中文测试音频能识别出非空中文文本。"""
    from pinecone.speech.asr.sensevoice import SenseVoiceRecognizer

    with wave.open(str(ASR_DIR / "test_wavs" / "zh.wav")) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

    text = await SenseVoiceRecognizer(AsrConfig(), ASR_DIR).transcribe(audio)
    assert any("\u4e00" <= ch <= "\u9fff" for ch in text)
