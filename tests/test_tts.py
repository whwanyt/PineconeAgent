"""TTS 分句、降级与 Speaker 流水线测试（使用假引擎，不需要网络和模型）。"""

import asyncio
import contextlib

import numpy as np
import pytest

from pinecone.audio.output import AudioOutput
from pinecone.core.config import PROJECT_ROOT, AudioConfig, TtsConfig
from pinecone.speech.tts.base import Audio, FallbackTTS, TextToSpeech
from pinecone.speech.tts.speaker import Speaker
from pinecone.speech.tts.text import SentenceSplitter, clean_for_speech, split_sentences


def test_split_and_clean() -> None:
    """按句末标点切分，去掉 Markdown 符号，纯标点不输出。"""
    assert split_sentences("**你好**！今天天气不错。……") == ["你好！", "今天天气不错。"]
    assert clean_for_speech("看 [这里](http://x) 和 `代码`") == "看 这里 和 代码"


def test_streaming_splitter_soft_break() -> None:
    """长句在逗号处提前切分，短句不切。"""
    splitter = SentenceSplitter()
    assert splitter.feed("好的，") == []
    long = "这是一个非常非常长的句子用来测试软切分逻辑是否生效，"
    assert splitter.feed(long) == ["好的，" + long]
    assert splitter.feed("结束") == [] and splitter.flush() == ["结束"]


class FakeTTS(TextToSpeech):
    name = "fake"

    def __init__(self, fail: bool = False, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> Audio:
        self.calls.append(text)
        await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("断网")
        return np.full(160, 0.5, np.float32), 16000


async def test_fallback_and_cache() -> None:
    """主引擎失败时使用降级引擎；短句命中缓存不再合成。"""
    primary, backup = FakeTTS(fail=True), FakeTTS()
    tts = FallbackTTS(primary, backup, cache_size=8)
    await tts.synthesize("好的")
    await tts.synthesize("好的")
    assert primary.calls == ["好的"] and backup.calls == ["好的"]

    with pytest.raises(ConnectionError):
        await FallbackTTS(FakeTTS(fail=True), None, 0).synthesize("x")


async def test_fallback_cooldown() -> None:
    """主引擎失败后，冷却期内不再尝试主引擎，冷却结束后恢复。"""
    primary, backup = FakeTTS(fail=True), FakeTTS()
    tts = FallbackTTS(primary, backup, cache_size=0, cooldown_s=0.05)
    await tts.synthesize("一")
    await tts.synthesize("二")
    assert primary.calls == ["一"] and backup.calls == ["一", "二"]
    await asyncio.sleep(0.06)
    primary.fail = False
    await tts.synthesize("三")
    assert primary.calls == ["一", "三"]


@pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "tts" / "model-steps-3.onnx").exists(), reason="未下载 tts 模型"
)
async def test_local_tts() -> None:
    """本地 Matcha 模型能合成出有声音的音频。"""
    from pinecone.speech.tts.local import LocalTTS

    samples, rate = await LocalTTS(TtsConfig(), PROJECT_ROOT / "models" / "tts").synthesize(
        "现在是3点20分。"
    )
    assert rate > 0 and samples.size > rate * 0.5 and np.abs(samples).max() > 0.05


@contextlib.asynccontextmanager
async def fake_sound_card(out: AudioOutput):
    """模拟声卡：持续从输出器拉取数据。"""

    async def drain() -> None:
        while True:
            out.mix(160)
            await asyncio.sleep(0.001)

    task = asyncio.create_task(drain())
    try:
        yield
    finally:
        task.cancel()


async def test_speaker_stream_and_cancel() -> None:
    """流式文本被分句合成并播完；播报任务被取消时立即停止。"""
    out = AudioOutput(AudioConfig())
    out.sample_rate = 16000
    speaker = Speaker(FakeTTS(), out)

    async def tokens():
        for piece in ["你好", "。今天", "不错！"]:
            yield piece

    async with fake_sound_card(out):
        text = await asyncio.wait_for(speaker.say_stream(tokens()), 2)
        assert text == "你好。今天不错！"

        slow = Speaker(FakeTTS(delay=10), out)
        task = asyncio.create_task(slow.say("这句话合成很慢。"))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled() and not out.is_playing
