"""播放器抽象与 mpv 实现。

为什么用 mpv（python-mpv 绑定 libmpv）：
    - 支持几乎所有格式和网络流（http / m3u8），树莓派上 CPU 占用很低；
    - 自带播放列表、暂停、音量、切歌；
    - 跑在自己的线程里，不阻塞事件循环。

注意：mpv 直接输出到声卡，不经过我们的 AudioOutput，因此软件 AEC 拿不到音乐的参考信号，
音乐响时唤醒率会下降。缓解方式：唤醒/播报时自动压低音乐音量（ducking，本技能已实现），
或使用 PipeWire 回声消除 / 硬件 AEC（见 config.yaml → audio.aec）。

安装：``sudo apt install libmpv-dev``（树莓派）/ ``brew install mpv``（macOS）。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .library import Track

logger = logging.getLogger(__name__)


class MusicPlayer(ABC):
    """播放器接口。所有方法都是非阻塞的，可以在事件循环线程中直接调用。"""

    @abstractmethod
    def play(self, tracks: list[Track]) -> None:
        """替换播放列表并从第一首开始播放。"""

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def resume(self) -> None: ...

    @abstractmethod
    def next(self) -> bool:
        """下一首，没有下一首时返回 False。"""

    @abstractmethod
    def previous(self) -> bool:
        """上一首，没有上一首时返回 False。"""

    @abstractmethod
    def stop(self) -> None:
        """停止并清空播放列表。"""

    @abstractmethod
    def set_volume(self, volume: int) -> None:
        """设置实际输出音量（0~100）。"""

    @property
    @abstractmethod
    def active(self) -> bool:
        """是否有播放列表在播（含暂停中）。播放列表播完后变为 False。"""

    @property
    @abstractmethod
    def paused(self) -> bool: ...

    @property
    @abstractmethod
    def current(self) -> Track | None:
        """当前曲目。"""

    def close(self) -> None:  # noqa: B027 - 可选钩子
        """释放资源。"""


class MpvPlayer(MusicPlayer):
    """基于 libmpv 的播放器。"""

    def __init__(self, audio_device: str | None = None) -> None:
        import mpv  # 延迟导入：没有 libmpv 时只是音乐技能不可用，不影响其他功能

        options = {
            "video": False,
            "ytdl": False,
            "input_default_bindings": False,
            "keep_open": False,
        }
        if audio_device:
            options["audio_device"] = audio_device
        self._mpv = mpv.MPV(**options)
        self._tracks: list[Track] = []

    def play(self, tracks: list[Track]) -> None:
        self._tracks = list(tracks)
        # 先清掉除当前曲目外的列表，再用 replace 替换当前曲目，切换时不会出现空闲间隙
        self._mpv.playlist_clear()
        for i, track in enumerate(tracks):
            self._mpv.loadfile(track.uri, "replace" if i == 0 else "append")
        self._mpv.pause = False

    def pause(self) -> None:
        self._mpv.pause = True

    def resume(self) -> None:
        self._mpv.pause = False

    def next(self) -> bool:
        pos = self._mpv.playlist_pos
        if pos is None or pos + 1 >= len(self._tracks):
            return False
        self._mpv.playlist_next("force")
        self._mpv.pause = False
        return True

    def previous(self) -> bool:
        pos = self._mpv.playlist_pos
        if not pos:
            return False
        self._mpv.playlist_prev("force")
        self._mpv.pause = False
        return True

    def stop(self) -> None:
        self._tracks = []
        self._mpv.stop()
        self._mpv.playlist_clear()

    def set_volume(self, volume: int) -> None:
        self._mpv.volume = max(0, min(100, volume))

    @property
    def active(self) -> bool:
        # idle_active：播放列表播完 / 被 stop 后为 True
        return bool(self._tracks) and not self._mpv.idle_active

    @property
    def paused(self) -> bool:
        return bool(self._mpv.pause)

    @property
    def current(self) -> Track | None:
        pos = self._mpv.playlist_pos
        if pos is None or not 0 <= pos < len(self._tracks):
            return None
        return self._tracks[pos]

    def close(self) -> None:
        self._mpv.terminate()


def create_player(audio_device: str | None = None) -> MusicPlayer:
    """创建播放器（目前只有 mpv）。"""
    return MpvPlayer(audio_device)
