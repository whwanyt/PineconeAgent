"""回声消除的单元测试。"""

import numpy as np
import pytest

from pinecone.audio.aec import create_echo_canceller
from pinecone.audio.aec.software import SoftwareEchoCanceller
from pinecone.core.config import AecConfig, AudioConfig


@pytest.mark.parametrize(
    ("mode", "effective"), [("none", False), ("hardware", True), ("pipewire", True)]
)
def test_factory_modes(mode: str, effective: bool) -> None:
    """工厂函数按配置返回对应模式，effective 标志正确。"""
    aec = create_echo_canceller(AudioConfig(aec=AecConfig(mode=mode)))
    assert aec.mode == mode and aec.effective is effective


def test_software_reference_padding() -> None:
    """参考缓冲不足时用 0 补齐，取出的样本与写入顺序一致。"""
    aec = SoftwareEchoCanceller(AudioConfig())
    aec._on_reference(np.array([0.1, 0.2], np.float32), 16000)
    far = aec._take_reference(4)
    assert np.allclose(far, [0.1, 0.2, 0.0, 0.0])


def test_software_webrtc_reduces_echo() -> None:
    """模拟扬声器回声（延迟 + 衰减），WebRTC 后端收敛后显著降低回声能量。"""
    pytest.importorskip("livekit")
    cfg = AudioConfig(aec=AecConfig(mode="software", backend="webrtc", noise_suppression=False))
    aec = SoftwareEchoCanceller(cfg)
    assert aec.effective

    rng = np.random.default_rng(0)
    frame = 320
    delay = 16 * 30  # 30ms 回声延迟
    far_all = (rng.standard_normal(16000 * 6) * 0.2).astype(np.float32)
    near_all = np.zeros_like(far_all)
    near_all[delay:] = far_all[:-delay] * 0.5

    in_energy, out_energy = [], []
    for start in range(0, far_all.size - frame + 1, frame):
        aec._on_reference(far_all[start : start + frame], 16000)
        near = near_all[start : start + frame]
        out = aec.process(near)
        if start > 16000 * 4:  # 只统计收敛之后的最后两秒
            in_energy.append(float(np.mean(near**2)))
            out_energy.append(float(np.mean(out**2)))

    reduction_db = 10 * np.log10(np.mean(in_energy) / max(np.mean(out_energy), 1e-12))
    assert reduction_db > 10, f"回声只降低了 {reduction_db:.1f}dB"
