"""PipeWire 系统级回声消除（aec.mode = pipewire）。

原理：PipeWire 的 ``module-echo-cancel`` 会创建一对虚拟设备：
    - ``Echo-Cancel Sink``：所有程序（包括 mpv 音乐）往这里放音，模块拿到完整参考信号；
    - ``Echo-Cancel Source``：去除回声后的麦克风信号。
因此它能消除 **音乐** 的回声，这是进程内 software 模式做不到的（音乐由 mpv 独立播放）。

配置方法：
    1. 运行 ``scripts/install_pi.sh``（会写入 ``~/.config/pipewire/pipewire.conf.d/echo-cancel.conf``）；
    2. config.yaml 中设置 ``input_device: "Echo-Cancel Source"``、``output_device: "Echo-Cancel Sink"``；
       或者把这两个虚拟设备设为系统默认设备。

程序内不做处理，只在接入时检查设备名称，配置不正确时给出警告。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pinecone.audio.aec.base import EchoCanceller

if TYPE_CHECKING:
    from pinecone.audio.input import AudioInput
    from pinecone.audio.output import AudioOutput
    from pinecone.core.config import AudioConfig

logger = logging.getLogger(__name__)


class PipeWireEchoCanceller(EchoCanceller):
    """PipeWire AEC：程序内只做配置检查。"""

    mode = "pipewire"

    def __init__(self, config: AudioConfig) -> None:
        self._config = config

    @property
    def effective(self) -> bool:
        return True

    def attach(self, mic: AudioInput, speaker: AudioOutput) -> None:
        mic.set_processor(None)
        for name, spec in (
            ("input_device", self._config.input_device),
            ("output_device", self._config.output_device),
        ):
            # 设为 null 时可能是用户已把虚拟设备设为系统默认，这里无法判断，只做提示
            if not (isinstance(spec, str) and "echo" in spec.lower()):
                logger.warning(
                    "aec.mode=pipewire，但 audio.%s=%r 不是 Echo-Cancel 虚拟设备；"
                    "除非已将其设为系统默认设备，否则回声消除不会生效",
                    name,
                    spec,
                )
