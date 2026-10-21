"""音频设备枚举与选择。

树莓派上 USB 麦克风、USB 声卡的设备序号会因为插拔顺序变化，
所以配置里推荐写设备名关键字（例如 ``"USB"``、``"ReSpeaker"``），由这里模糊匹配成序号。

调试时可以执行 ``python -m pinecone.audio.devices`` 列出所有设备。
"""

from __future__ import annotations

import logging
from typing import Literal

import sounddevice as sd

logger = logging.getLogger(__name__)

Direction = Literal["input", "output"]


def list_devices(direction: Direction | None = None) -> list[dict]:
    """列出音频设备。

    Args:
        direction: ``input`` 只列麦克风，``output`` 只列扬声器，``None`` 全部列出。

    Returns:
        设备信息字典列表，额外包含 ``index`` 字段。
    """
    devices = []
    for index, info in enumerate(sd.query_devices()):
        info = dict(info)
        info["index"] = index
        if direction == "input" and info["max_input_channels"] <= 0:
            continue
        if direction == "output" and info["max_output_channels"] <= 0:
            continue
        devices.append(info)
    return devices


def resolve_device(spec: str | int | None, direction: Direction) -> int | None:
    """把配置中的设备描述解析成 sounddevice 可用的设备序号。

    Args:
        spec: ``None``（系统默认）/ 整数序号 / 设备名关键字（不区分大小写）。
        direction: 设备方向。

    Returns:
        设备序号；``None`` 表示使用系统默认设备。

    Raises:
        ValueError: 按关键字找不到匹配的设备。
    """
    if spec is None or isinstance(spec, int):
        return spec
    if spec.isdigit():
        return int(spec)
    keyword = spec.lower()
    for info in list_devices(direction):
        if keyword in info["name"].lower():
            logger.info(
                "音频%s设备：[%d] %s",
                "输入" if direction == "input" else "输出",
                info["index"],
                info["name"],
            )
            return info["index"]
    names = ", ".join(d["name"] for d in list_devices(direction))
    raise ValueError(f"找不到名称包含「{spec}」的{direction}设备，可用设备：{names}")


def device_info(device: int | None, direction: Direction) -> dict:
    """查询设备信息（采样率、声道数等）。``None`` 表示默认设备。"""
    return dict(sd.query_devices(device, direction))


if __name__ == "__main__":
    # 命令行工具：列出所有音频设备，方便填写 config.yaml 中的设备关键字
    for d in list_devices():
        kinds = []
        if d["max_input_channels"] > 0:
            kinds.append(f"输入{d['max_input_channels']}声道")
        if d["max_output_channels"] > 0:
            kinds.append(f"输出{d['max_output_channels']}声道")
        print(
            f"[{d['index']:2d}] {d['name']}  ({'，'.join(kinds)}，默认 {d['default_samplerate']:.0f}Hz)"
        )
