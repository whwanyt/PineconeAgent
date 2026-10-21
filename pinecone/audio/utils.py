"""音频小工具：重采样、格式转换、提示音生成。

这些函数都是纯 numpy 运算，不依赖声卡，便于单元测试。
"""

from __future__ import annotations

import numpy as np


def to_mono(samples: np.ndarray) -> np.ndarray:
    """多声道转单声道（取各声道平均值）。已是单声道时原样返回。"""
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性插值重采样。

    为什么不用高质量的多相滤波？
        这里只用于语音（采集 48k -> 16k、TTS 24k -> 48k 播放），线性插值对语音可懂度
        影响很小，而且零依赖、在树莓派上几乎不占 CPU。如果以后要播放高保真音乐，
        音乐由 mpv 自己处理，不经过这里。

    Args:
        samples: 单声道 float32 样本。
        src_rate: 原采样率。
        dst_rate: 目标采样率。

    Returns:
        重采样后的 float32 样本。
    """
    samples = np.asarray(samples, dtype=np.float32)
    if src_rate == dst_rate or samples.size == 0:
        return samples
    n_out = int(round(samples.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(samples.size, dtype=np.float64)
    x_new = np.linspace(0, samples.size - 1, n_out)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def int16_to_float(pcm: np.ndarray | bytes) -> np.ndarray:
    """16 位整型 PCM 转 float32 [-1, 1]。"""
    if isinstance(pcm, bytes | bytearray):
        pcm = np.frombuffer(pcm, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def float_to_int16(samples: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] 转 16 位整型 PCM（超出范围的值会被截断）。"""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def beep(
    freq: float = 880.0,
    duration: float = 0.12,
    sample_rate: int = 16000,
    volume: float = 0.3,
) -> np.ndarray:
    """生成一个带淡入淡出的正弦提示音（唤醒成功时播放）。

    淡入淡出各 10ms，避免波形突变产生“咔哒”声。
    """
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    tone = np.sin(2 * np.pi * freq * t) * volume
    fade = min(n // 2, int(sample_rate * 0.01))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
    return tone.astype(np.float32)
