"""在线歌曲的参数、列表解析和播放地址缓存。

搜索、歌单、专辑、榜单返回的是歌曲信息，不是可以长期保存的地址。
音乐技能把信息放进播放队列，快轮到时再通过 :class:`UrlResolver` 调用配置的取地址工具。

地址只放在内存里，十分钟后失效。同一时刻只允许一个取地址或取列表的请求，
避免和正在进行的另一次音乐服务调用抢同一个会话。

这些函数不碰播放器，可以在没有网络的测试里跑。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .library import Track, search_key

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 600
_LIST_KEYS = ("songs", "tracks", "list", "data", "items")
# 封面、网页和播放地址都不进歌单。地址会过期，封面和歌词对播放没有用。
_DROP_FIELDS = {"cover", "link", "lyric", "lyrics", "url", "uri"}

CallTool = Callable[[str, dict[str, Any]], Awaitable[str]]


class ResolveUnavailable(Exception):
    """取播放地址或取歌单的工具没有连上、超时，或没有返回地址。"""


class SongArg(BaseModel):
    """一首在线歌，字段照着音乐 MCP 的歌曲对象。

    ``id`` 有时是数字，这里收成字符串。``extra`` 里的值也收成字符串，
    因为取地址服务要求 ``extra`` 的值是字符串。时长、大小、码率保持数字，
    ``is_vip`` 保持布尔，否则服务端解不出这些字段。
    """

    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

    id: str = ""
    name: str = ""
    artist: str = ""
    album: str = ""
    album_id: str = ""
    duration: int | None = None
    size: int | None = None
    bitrate: int | None = None
    source: str = ""
    ext: str = ""
    is_vip: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
    uri: str = ""


def song_payload(song: SongArg) -> dict[str, Any]:
    """做成可以存进歌单、也可以交回取地址工具的歌曲对象。"""
    payload: dict[str, Any] = {}
    for key, value in song.model_dump().items():
        if key in _DROP_FIELDS or value in (None, "", {}, []):
            continue
        if key == "extra" and isinstance(value, dict):
            extra = {
                str(name): _as_text(item) for name, item in value.items() if item not in (None, "")
            }
            if extra:
                payload["extra"] = extra
            continue
        payload[key] = value
    if not payload.get("id") or not payload.get("source"):
        return {}
    return payload


def track_from_song(item: SongArg | dict[str, Any]) -> Track | None:
    """把工具参数里的一首歌变成曲目。

    既接受 pydantic 对象，也接受字典。音乐框架会把嵌套参数按模型传进来。
    带 http(s) 地址时曲目可以直接播，歌曲信息仍保留下来，方便地址过期后重取。
    """
    try:
        song = item if isinstance(item, SongArg) else SongArg.model_validate(item)
    except ValidationError:
        logger.info("无法识别的歌曲参数：%s", item)
        return None
    payload = song_payload(song)
    stored = json.dumps(payload, ensure_ascii=False) if payload else ""
    uri = song.uri.strip()
    playable = uri if _is_http(uri) or _is_local(uri) else ""
    title = song.name or _title_from_uri(playable)
    if not stored and not playable:
        return None
    return Track(
        title=title or "在线音乐",
        uri=playable,
        search_key=search_key(f"{song.artist}{title}"),
        artist=song.artist,
        song=stored,
    )


def songs_from_result(text: str) -> list[dict[str, Any]]:
    """从取歌工具的文本结果里拿出歌曲对象。

    接受顶层就是列表，或包在 ``songs`` / ``tracks`` / ``list`` / ``data`` / ``items`` 里。
    只向下看一层。解析失败时记下原文并返回空列表。
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.info("取歌结果不是 JSON：%s", text[:300])
        return []
    rows = _song_rows(data)
    songs = [row for row in rows if isinstance(row, dict)]
    if not songs:
        logger.info("取歌结果里没有歌曲：%s", text[:300])
    return songs


def parse_audio_url(text: str) -> str | None:
    """从取地址结果里拿出 http(s) 地址。

    整段文本就是地址，或 JSON 的 ``url`` / ``data.url``。
    不在错误说明里搜地址，避免把报错里的链接当成可以播放的歌。
    """
    stripped = text.strip()
    if _is_http(stripped) and "\n" not in stripped:
        return stripped
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    url = data.get("url")
    if _is_http(url):
        return url
    inner = data.get("data")
    if isinstance(inner, dict) and _is_http(inner.get("url")):
        return str(inner["url"])
    return None


class UrlResolver:
    """按需取播放地址，并缓存一小会儿。

    ``call_tool`` 由技能上下文提供。工具名来自配置 ``url_tool``。
    只在事件循环线程使用。``resolve`` 会等待网络，超时或取消都会抛出，
    ``CancelledError`` 原样传播。
    """

    def __init__(self, call_tool: CallTool, tool: str, timeout_s: float) -> None:
        self._call_tool = call_tool
        self._tool = tool
        self._timeout_s = timeout_s
        self._cache: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    def remember(self, track: Track, uri: str) -> None:
        """把已经拿到的地址放进缓存。搜索结果里偶尔会自带地址。"""
        if uri:
            self._cache[track.key] = (uri, time.monotonic() + _CACHE_TTL_S)

    def cached(self, track: Track) -> str | None:
        """还没过期的地址。没有时返回 None。"""
        item = self._cache.get(track.key)
        if item is None or item[1] <= time.monotonic():
            return None
        return item[0]

    async def resolve(self, track: Track, *, fresh: bool = False) -> str:
        """得到可以交给播放器的地址。

        本地文件和电台直接用曲目自己的地址。在线歌先看缓存；
        ``fresh`` 为真时跳过缓存，再向音乐服务要一次。

        Raises:
            ResolveUnavailable: 没配置工具、工具不在、超时，或结果里没有地址。
        """
        if track.uri and not fresh:
            return track.uri
        if not fresh:
            cached = self.cached(track)
            if cached:
                return cached
        if not self._tool:
            raise ResolveUnavailable("没有配置取地址工具")
        if not track.song_info():
            raise ResolveUnavailable(f"没有{track.label}的歌曲信息")
        text = await self._invoke(self._tool, {"song": track.song_info()})
        url = parse_audio_url(text)
        if url is None:
            raise ResolveUnavailable(f"没有拿到{track.label}的播放地址")
        self.remember(track, url)
        return url

    async def fetch_list(self, tool: str, args: dict[str, Any]) -> str:
        """调用白名单里的取歌工具，返回原始文本。

        Raises:
            ResolveUnavailable: 超时或音乐服务没有连上。
        """
        return await self._invoke(tool, args)

    async def _invoke(self, tool: str, args: dict[str, Any]) -> str:
        """同一时刻只跑一个音乐服务调用。"""
        async with self._lock:
            try:
                async with asyncio.timeout(self._timeout_s):
                    text = await self._call_tool(tool, args)
            except TimeoutError as exc:
                raise ResolveUnavailable("音乐服务没有响应") from exc
            except LookupError as exc:
                raise ResolveUnavailable("音乐服务没有连上") from exc
        if "暂时不可用" in text:
            raise ResolveUnavailable("音乐服务没有响应")
        return text


def _song_rows(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in _LIST_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for inner_key in _LIST_KEYS:
                inner = value.get(inner_key)
                if isinstance(inner, list):
                    return inner
    return []


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)


def _is_http(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _is_local(uri: str) -> bool:
    return uri.startswith("~") or uri.startswith("/")


def _title_from_uri(uri: str) -> str:
    name = uri.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name
