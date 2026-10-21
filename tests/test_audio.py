"""音频工具与播放混音逻辑的单元测试（不需要声卡）。"""

import asyncio

import numpy as np

from pinecone.audio.output import AudioOutput
from pinecone.audio.utils import beep, float_to_int16, int16_to_float, resample
from pinecone.core.config import AudioConfig


def test_resample_length_and_identity() -> None:
    """重采样后长度按比例变化；同采样率原样返回。"""
    x = np.ones(480, dtype=np.float32)
    assert resample(x, 48000, 16000).size == 160
    assert resample(x, 16000, 16000) is x


def test_int16_roundtrip() -> None:
    """int16 与 float 互转误差很小。"""
    x = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    assert np.allclose(int16_to_float(float_to_int16(x)), x, atol=1e-4)


def test_beep_has_fade() -> None:
    """提示音首尾为淡入淡出，接近 0。"""
    tone = beep(sample_rate=16000)
    assert abs(tone[0]) < 1e-3 and abs(tone[-1]) < 0.01


def make_output() -> AudioOutput:
    out = AudioOutput(AudioConfig(volume=1.0))
    out.sample_rate = 16000  # 不打开声卡，直接指定采样率
    return out


async def test_mix_plays_handles_in_order() -> None:
    """多个句柄按顺序播放，播完后 wait 返回。"""
    out = make_output()
    a, b = out.open(), out.open()
    a.feed(np.full(100, 0.1, np.float32), 16000)
    a.finish()
    b.feed(np.full(100, 0.2, np.float32), 16000)
    b.finish()

    mixed = out.mix(250)
    assert np.allclose(mixed[:100], 0.1) and np.allclose(mixed[100:200], 0.2)
    assert np.allclose(mixed[200:], 0.0)
    await asyncio.wait_for(asyncio.gather(a.wait(), b.wait()), 1)
    assert not out.is_playing


async def test_streaming_underrun_waits_for_more_data() -> None:
    """流式句柄数据不足时输出静音，不会跳到后面的句柄。"""
    out = make_output()
    stream, later = out.open(), out.open()
    later.feed(np.full(10, 0.9, np.float32), 16000)
    later.finish()
    stream.feed(np.full(10, 0.1, np.float32), 16000)

    mixed = out.mix(30)
    assert np.allclose(mixed[:10], 0.1) and np.allclose(mixed[10:], 0.0)
    assert not stream.done


async def test_cancel_wait_stops_playback() -> None:
    """等待播放的任务被取消时，句柄立即停止。"""
    out = make_output()
    task = asyncio.create_task(out.play(np.ones(16000 * 10, np.float32), 16000))
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert np.allclose(out.mix(100), 0.0)
    assert not out.is_playing
