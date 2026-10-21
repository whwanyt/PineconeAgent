"""不做回声消除（aec.mode = none）。

适合初期没有任何额外设备、或者软件 AEC 效果不理想时使用。
此时只能通过唤醒词打断，并且播报期间会提高唤醒阈值，降低被自己的声音误唤醒的概率。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pinecone.audio.aec.base import EchoCanceller

if TYPE_CHECKING:
    from pinecone.audio.input import AudioInput
    from pinecone.audio.output import AudioOutput


class NoEchoCanceller(EchoCanceller):
    """空实现。"""

    mode = "none"

    @property
    def effective(self) -> bool:
        return False

    def attach(self, mic: AudioInput, speaker: AudioOutput) -> None:
        mic.set_processor(None)
