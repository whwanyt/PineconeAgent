"""回声消除接口与工厂函数。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pinecone.audio.input import AudioInput
    from pinecone.audio.output import AudioOutput
    from pinecone.core.config import AudioConfig


class EchoCanceller(ABC):
    """回声消除器接口。"""

    mode: str = ""
    """对应的配置模式名。"""

    @property
    @abstractmethod
    def effective(self) -> bool:
        """回声消除是否真正生效。

        打断逻辑据此决定能否启用“直接开口打断”（VAD 打断）：
        没有回声消除时，扬声器的声音会被 VAD 当成用户说话，导致自己打断自己。
        """

    @abstractmethod
    def attach(self, mic: AudioInput, speaker: AudioOutput) -> None:
        """接入音频链路（注册参考信号监听、设置麦克风帧处理器等）。"""

    def close(self) -> None:  # noqa: B027 - 默认无资源需要释放
        """释放资源。"""


def create_echo_canceller(config: AudioConfig) -> EchoCanceller:
    """根据配置创建回声消除器。

    Args:
        config: 音频配置（使用其中的 ``aec`` 段）。

    Returns:
        对应模式的回声消除器实例。software 模式下如果没有可用算法库，会自动降级为不处理，
        此时 ``effective`` 为 ``False``。
    """
    # 延迟导入：只有用到的模式才导入对应依赖
    mode = config.aec.mode
    if mode == "software":
        from pinecone.audio.aec.software import SoftwareEchoCanceller

        return SoftwareEchoCanceller(config)
    if mode == "pipewire":
        from pinecone.audio.aec.pipewire import PipeWireEchoCanceller

        return PipeWireEchoCanceller(config)
    if mode == "hardware":
        from pinecone.audio.aec.hardware import HardwareEchoCanceller

        return HardwareEchoCanceller()
    from pinecone.audio.aec.none import NoEchoCanceller

    return NoEchoCanceller()
