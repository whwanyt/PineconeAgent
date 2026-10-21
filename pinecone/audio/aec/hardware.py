"""硬件回声消除（aec.mode = hardware）。

适用于自带 AEC 芯片的麦克风阵列，例如 ReSpeaker USB Mic Array v2.0（XVF3000）。
注意：硬件 AEC 需要拿到扬声器的回采信号，所以扬声器必须接在阵列自带的 3.5mm 输出口上，
并在 config.yaml 中把 ``output_device`` 也设为该阵列。

程序内不做任何处理，只把 ``effective`` 声明为 True，允许启用 VAD 打断。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pinecone.audio.aec.base import EchoCanceller

if TYPE_CHECKING:
    from pinecone.audio.input import AudioInput
    from pinecone.audio.output import AudioOutput


class HardwareEchoCanceller(EchoCanceller):
    """硬件 AEC：程序内不处理。"""

    mode = "hardware"

    @property
    def effective(self) -> bool:
        return True

    def attach(self, mic: AudioInput, speaker: AudioOutput) -> None:
        mic.set_processor(None)
