"""本地曲库与电台。

曲库就是一个（或多个）音乐目录，按文件名和所在目录名搜索，因此推荐按
``歌手/专辑/歌名.mp3`` 或 ``歌手 - 歌名.mp3`` 组织文件，就能用“播放周杰伦的歌”“播放晴天”点歌。

电台是“名称 → 流地址”的映射（config.yaml 中配置），例如::

    stations:
      中国之声: http://ngcdn001.cnr.cn/live/zgzs/index.m3u8
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pinecone.nlu.matcher import normalize

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aac", ".opus", ".ape"}


def search_key(text: str) -> str:
    """搜索用的归一化：去标点、空白和常见分隔符（“陈奕迅 - 十年” → “陈奕迅十年”）。"""
    return re.sub(r"[\s\-_.&]+", "", normalize(text))


@dataclass(frozen=True)
class Track:
    """一首歌（或一个电台流）。

    在线歌曲可以暂时没有 ``uri``：播放列表和自建歌单只保存 ``song`` 里的歌曲信息，
    快轮到时再取播放地址。``uri`` 有值时优先当作已经可以交给播放器的地址。
    """

    title: str
    """播报用的名称（文件名去掉扩展名，或电台名、在线歌名）。"""
    uri: str
    """文件路径或流地址。在线歌还没取地址时为空。"""
    search_key: str = ""
    """搜索用的归一化文本（含目录名，便于按歌手搜索）。"""
    artist: str = ""
    """歌手。在线歌曲用来组成「黄诗扶的九万字」这样的播报名。"""
    song: str = ""
    """在线歌曲信息的 JSON。不含封面、歌词和会过期的播放地址。"""

    @property
    def online(self) -> bool:
        """是否是还没落到本地文件的在线歌曲。"""
        return not self.uri and bool(self.song)

    @property
    def key(self) -> str:
        """去重用的稳定标识。有地址用地址，否则用 ``来源:id``。"""
        if self.uri:
            return self.uri
        info = self.song_info()
        source = str(info.get("source") or "")
        song_id = str(info.get("id") or "")
        if source or song_id:
            return f"{source}:{song_id}"
        return self.search_key or self.title

    @property
    def label(self) -> str:
        """适合朗读的名字。有歌手时带上歌手。"""
        if self.artist:
            return f"{self.artist}的{self.title}"
        return self.title

    def song_info(self) -> dict[str, Any]:
        """还原取播放地址需要的歌曲对象。没有在线信息时返回空字典。"""
        if not self.song:
            return {}
        try:
            data = json.loads(self.song)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


class MusicLibrary:
    """扫描音乐目录并提供搜索。"""

    def __init__(self, folders: list[str | Path], stations: dict[str, str] | None = None) -> None:
        self._folders = [Path(f).expanduser() for f in folders]
        self._stations = [
            Track(title=name, uri=url, search_key=search_key(name))
            for name, url in (stations or {}).items()
        ]
        self._tracks: list[Track] | None = None

    @property
    def tracks(self) -> list[Track]:
        """全部本地歌曲（首次访问时扫描）。"""
        if self._tracks is None:
            self.rescan()
        assert self._tracks is not None
        return self._tracks

    def rescan(self) -> int:
        """重新扫描音乐目录，返回歌曲数量。"""
        tracks: list[Track] = []
        for folder in self._folders:
            if not folder.is_dir():
                logger.warning("音乐目录不存在：%s", folder)
                continue
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
                    continue
                relative = path.relative_to(folder).with_suffix("")
                tracks.append(
                    Track(
                        title=path.stem,
                        uri=str(path),
                        search_key=search_key("".join(relative.parts)),
                    )
                )
        self._tracks = tracks
        logger.info("曲库扫描完成：%d 首", len(tracks))
        return len(tracks)

    def find_station(self, query: str) -> Track | None:
        """按名称找电台。"""
        key = search_key(query)
        if not key:
            return None
        for station in self._stations:
            if key in station.search_key or station.search_key in key:
                return station
        return None

    def search(self, query: str, *, shuffle: bool = True) -> list[Track]:
        """搜索歌曲。

        - 空查询：返回全部歌曲；
        - 先找歌名完全相同的，再找“目录名 + 文件名”包含关键词的，
          因此“周杰伦晴天”“陈奕迅十年”这种歌手 + 歌名的说法也能命中。
        """
        key = search_key(query)
        tracks = list(self.tracks)
        if key:
            exact = [t for t in tracks if search_key(t.title) == key]
            contains = [t for t in tracks if t not in exact and key in t.search_key]
            if shuffle:
                random.shuffle(contains)
            return exact + contains
        if shuffle:
            random.shuffle(tracks)
        return tracks
