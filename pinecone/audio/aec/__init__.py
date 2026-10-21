"""回声消除（AEC）。

按配置项 ``audio.aec.mode`` 选择实现，四种模式对外接口一致（:class:`EchoCanceller`）::

    none      → NoEchoCanceller        不处理
    software  → SoftwareEchoCanceller  进程内 WebRTC APM / speexdsp
    pipewire  → PipeWireEchoCanceller  系统级处理，程序内只做配置检查
    hardware  → HardwareEchoCanceller  麦克风阵列芯片处理，程序内不做处理

调用方只需要：``aec = create_echo_canceller(cfg.audio); aec.attach(mic, speaker)``。
"""

from pinecone.audio.aec.base import EchoCanceller, create_echo_canceller

__all__ = ["EchoCanceller", "create_echo_canceller"]
