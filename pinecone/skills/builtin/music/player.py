"""播放器抽象与 mpv 实现。

mpv 只负责正在听的这一首。播放列表、循环和随机在 :mod:`playqueue` 里。
一首结束时看 ``loadfile`` 返回的 ``playlist_entry_id``：被换掉的旧曲目是
ABORTED，id 不同，不会被当成播完。python-mpv 的 ``loadfile()`` 包装会丢掉这个
返回值，所以直接调用 ``command``。

注意：mpv 直接输出到声卡，不经过我们的 AudioOutput，因此软件 AEC 拿不到音乐的参考信号，
音乐响时唤醒率会下降。缓解方式：唤醒/播报时自动压低音乐音量（ducking，本技能已实现），
或使用 PipeWire 回声消除 / 硬件 AEC（见 config.yaml → audio.aec）。

安装：``sudo apt install libmpv-dev``（树莓派）/ ``brew install mpv``（macOS）。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from enum import StrEnum

logger = logging.getLogger(__name__)

# ctypes.macholib.dyld 在未设置 DYLD_FALLBACK_LIBRARY_PATH 时使用的默认目录。
# 该变量一旦被设置就会替换这些默认值，补 Homebrew 路径时必须把它们带上。
_DYLD_DEFAULT_LIBRARY_FALLBACK = ("~/lib", "/usr/local/lib", "/lib", "/usr/lib")
# Apple Silicon 在 /opt/homebrew，Intel Homebrew 在 /usr/local。
_HOMEBREW_LIB_CANDIDATES = ("/opt/homebrew/lib", "/usr/local/lib")
# 没有 entry id 时，连续多少次看到空闲才认为这一首结束。
_IDLE_POLLS = 2


class PlayState(StrEnum):
    """当前这一首的播放状态。暂停仍算正在播放。"""

    IDLE = "idle"
    PLAYING = "playing"
    ENDED = "ended"
    FAILED = "failed"


class MusicPlayer(ABC):
    """单曲播放器。方法非阻塞，只在事件循环线程调用，不涉及音频回调。"""

    @abstractmethod
    def load(self, uri: str) -> None:
        """换成这一首并开始播放。"""

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def resume(self) -> None: ...

    @abstractmethod
    def stop(self) -> None:
        """停止当前这一首。"""

    @abstractmethod
    def set_volume(self, volume: int) -> None:
        """设置实际输出音量（0~100）。"""

    @property
    @abstractmethod
    def state(self) -> PlayState:
        """当前状态。暂停时仍是 :attr:`PlayState.PLAYING`。"""

    @property
    @abstractmethod
    def paused(self) -> bool: ...

    def close(self) -> None:  # noqa: B027 - 可选钩子
        """释放资源。"""


def _prepare_libmpv_search_path() -> None:
    """让 macOS 上的 python-mpv 能找到 Homebrew 的 libmpv。

    ``ctypes.util.find_library`` 只搜索 dyld 默认路径（``/usr/lib`` 等）。
    Apple Silicon 的 ``brew install mpv`` 把 ``libmpv.dylib`` 放在
    ``/opt/homebrew/lib``，不在其中，因此必须在 ``import mpv`` 之前写入
    ``DYLD_FALLBACK_LIBRARY_PATH``。Linux 由 ldconfig 解析，这里什么都不做。

    只修改当前进程的环境变量，在创建播放器时于事件循环线程调用，不涉及音频回调。
    """
    if sys.platform != "darwin":
        return
    extras = [path for path in _HOMEBREW_LIB_CANDIDATES if os.path.isdir(path)]
    if not extras:
        return
    key = "DYLD_FALLBACK_LIBRARY_PATH"
    current = os.environ.get(key)
    if current:
        parts = [part for part in current.split(":") if part]
    else:
        parts = [os.path.expanduser(path) for path in _DYLD_DEFAULT_LIBRARY_FALLBACK]
    for path in reversed(extras):
        if path not in parts:
            parts.insert(0, path)
    os.environ[key] = ":".join(parts)


def _entry_id(result: object) -> int | None:
    """从 ``loadfile`` 的返回值里取出曲目编号。老版本 mpv 没有这个返回值。"""
    if isinstance(result, bool) or result is None:
        return None
    if isinstance(result, int) and result > 0:
        return result
    if isinstance(result, float) and result.is_integer() and result > 0:
        return int(result)
    if isinstance(result, str) and result.isdigit() and int(result) > 0:
        return int(result)
    return None


class MpvPlayer(MusicPlayer):
    """基于 libmpv 的单曲播放器。

    END_FILE 回调跑在 mpv 的事件线程里，只把结束原因写进 ``_ends``。
    事件循环线程读取时加同一把锁。回调里不做阻塞操作。
    """

    def __init__(self, audio_device: str | None = None) -> None:
        # 必须先补搜索路径再导入：python-mpv 在 import 时就用 find_library 加载 libmpv。
        _prepare_libmpv_search_path()
        import mpv  # 延迟导入：没有 libmpv 时只是音乐技能不可用，不影响其他功能

        options: dict[str, object] = {
            "video": False,
            "ytdl": False,
            "input_default_bindings": False,
            "keep_open": False,
            "network_timeout": 15,
        }
        if audio_device:
            options["audio_device"] = audio_device
        self._mpv = mpv.MPV(**options)
        self._eof = mpv.MpvEventEndFile.EOF
        self._error = mpv.MpvEventEndFile.ERROR
        self._redirect = mpv.MpvEventEndFile.REDIRECT
        self._lock = threading.Lock()
        self._ends: dict[int, bool] = {}
        self._entry_id: int | None = None
        self._state = PlayState.IDLE
        self._paused = False
        self._fallback = False
        self._idle_polls = 0

        @self._mpv.event_callback("end-file")
        def _on_end(event: object) -> None:
            self._remember_end(event)

        self._on_end = _on_end

    def load(self, uri: str) -> None:
        try:
            result = self._mpv.command("loadfile", uri, "replace")
        except Exception:
            logger.warning("加载音频失败：%s", uri, exc_info=True)
            with self._lock:
                self._state = PlayState.FAILED
                self._entry_id = None
                self._fallback = False
            return
        entry = _entry_id(result)
        with self._lock:
            self._entry_id = entry
            self._fallback = entry is None
            self._idle_polls = 0
            self._state = PlayState.PLAYING
            self._paused = False
        self._mpv.pause = False

    def pause(self) -> None:
        self._mpv.pause = True
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        self._mpv.pause = False
        with self._lock:
            self._paused = False

    def stop(self) -> None:
        self._mpv.stop()
        with self._lock:
            self._state = PlayState.IDLE
            self._entry_id = None
            self._paused = False
            self._fallback = False

    def set_volume(self, volume: int) -> None:
        self._mpv.volume = max(0, min(100, volume))

    @property
    def state(self) -> PlayState:
        if self._fallback:
            idle = bool(self._mpv.idle_active)
            with self._lock:
                return self._note_idle(idle)
        with self._lock:
            return self._note_entry()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def close(self) -> None:
        self._mpv.terminate()

    def _remember_end(self, event: object) -> None:
        """记下这一首是正常结束还是失败。电台跳转时改跟新的曲目编号。"""
        data = getattr(event, "data", None)
        if data is None:
            return
        reason = int(data.reason)
        entry = int(data.playlist_entry_id)
        with self._lock:
            if reason == self._redirect and self._entry_id == entry:
                self._entry_id = int(data.playlist_insert_id)
                return
            if reason == self._eof:
                self._ends[entry] = False
            elif reason == self._error:
                self._ends[entry] = True

    def _note_entry(self) -> PlayState:
        if self._state is not PlayState.PLAYING:
            return self._state
        entry = self._entry_id
        if entry is not None and entry in self._ends:
            failed = self._ends.pop(entry)
            self._state = PlayState.FAILED if failed else PlayState.ENDED
        return self._state

    def _note_idle(self, idle: bool) -> PlayState:
        """老版本 mpv 没有曲目编号时，用空闲状态去抖。"""
        if self._state is not PlayState.PLAYING:
            return self._state
        if not idle:
            self._idle_polls = 0
            return self._state
        self._idle_polls += 1
        if self._idle_polls >= _IDLE_POLLS:
            self._state = PlayState.ENDED
        return self._state


def create_player(audio_device: str | None = None) -> MusicPlayer:
    """创建播放器（目前只有 mpv）。"""
    return MpvPlayer(audio_device)
