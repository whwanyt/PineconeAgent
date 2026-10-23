"""在线歌曲参数、列表解析和地址缓存。不访问网络。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pinecone.skills.builtin.music.online import (
    ResolveUnavailable,
    SongArg,
    UrlResolver,
    parse_audio_url,
    preload_url,
    song_payload,
    songs_from_result,
    track_from_song,
)


def test_song_payload_keeps_numbers_and_stringifies_extra() -> None:
    """时长保持数字，extra 的值变成字符串，封面和过期地址不保存。"""
    song = SongArg.model_validate(
        {
            "id": 9,
            "source": "netease",
            "name": "九万字",
            "artist": "黄诗扶",
            "duration": 200,
            "is_vip": False,
            "extra": {"fee": 1},
            "cover": "http://example.com/cover.jpg",
            "url": "http://example.com/expired",
        }
    )
    payload = song_payload(song)
    assert payload["id"] == "9"
    assert payload["duration"] == 200
    assert payload["is_vip"] is False
    assert payload["extra"] == {"fee": "1"}
    assert "cover" not in payload
    assert "url" not in payload


def test_track_from_song_accepts_model_and_dict() -> None:
    """嵌套参数无论是模型还是字典，都能变成在线曲目。"""
    model = SongArg(id="9", source="netease", name="九万字", artist="黄诗扶")
    from_model = track_from_song(model)
    from_dict = track_from_song(
        {"id": "9", "source": "netease", "name": "九万字", "artist": "黄诗扶"}
    )
    assert from_model is not None and from_dict is not None
    assert from_model.online and from_model.key == "netease:9"
    assert from_dict.song_info()["artist"] == "黄诗扶"
    assert track_from_song({"name": "没有来源"}) is None


def test_track_from_song_keeps_http_uri() -> None:
    """有歌曲身份时地址不写进曲目；只有地址时直接播放。"""
    item = {
        "id": "9",
        "source": "netease",
        "name": "九万字",
        "uri": "https://example.com/a.mp3",
    }
    track = track_from_song(item)
    assert track is not None
    assert track.uri == ""
    assert track.online
    assert track.song_info()["id"] == "9"
    assert preload_url(item) == "https://example.com/a.mp3"

    bare = track_from_song({"name": "甲", "uri": "https://example.com/a.mp3"})
    assert bare is not None and bare.uri == "https://example.com/a.mp3"


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ('[{"id":"1","name":"甲"}]', 1),
        ('{"songs":[{"id":"1"}]}', 1),
        ('{"data":{"tracks":[{"id":"1"},{"id":"2"}]}}', 2),
        ("不是 json", 0),
        ('{"error":"没有"}', 0),
    ],
)
def test_songs_from_result(text: str, count: int) -> None:
    """取歌结果可以是顶层列表，或包在常见的列表字段里。"""
    assert len(songs_from_result(text)) == count


def test_parse_audio_url_ignores_links_inside_errors() -> None:
    """只认整段地址或 JSON 里的 url，不从报错句子里抠链接。"""
    assert parse_audio_url("https://example.com/a.mp3") == "https://example.com/a.mp3"
    assert parse_audio_url('{"url":"https://example.com/a.mp3"}') == "https://example.com/a.mp3"
    assert (
        parse_audio_url('{"data":{"url":"https://example.com/a.mp3"}}')
        == "https://example.com/a.mp3"
    )
    assert parse_audio_url("失败了，见 https://example.com/help") is None


def _track() -> Any:
    track = track_from_song({"id": "9", "source": "netease", "name": "九万字", "artist": "黄诗扶"})
    assert track is not None
    return track


async def test_resolver_caches_and_refetches_when_asked() -> None:
    """同一首第二次用缓存；要求重新取时才再调用工具。"""
    calls: list[str] = []

    async def call_json(name: str, args: dict[str, Any]) -> str:
        calls.append(args["song"]["id"])
        return '{"url":"https://example.com/' + str(len(calls)) + '.mp3"}'

    resolver = UrlResolver(call_json, "get_audio_url", timeout_s=1)
    track = _track()
    assert await resolver.resolve(track) == "https://example.com/1.mp3"
    assert await resolver.resolve(track) == "https://example.com/1.mp3"
    assert await resolver.resolve(track, fresh=True) == "https://example.com/2.mp3"
    assert calls == ["9", "9"]


async def test_resolver_treats_timeout_and_missing_tool_as_unavailable() -> None:
    """超时、工具不在、调用报错，以及「暂时不可用」都变成同一种失败。"""

    async def too_slow(name: str, args: dict[str, Any]) -> str:
        await asyncio.sleep(1)
        return '{"url":"https://example.com/a.mp3"}'

    resolver = UrlResolver(too_slow, "get_audio_url", timeout_s=0.01)
    with pytest.raises(ResolveUnavailable, match="没有响应"):
        await resolver.resolve(_track())

    async def missing(name: str, args: dict[str, Any]) -> str:
        raise LookupError(name)

    resolver = UrlResolver(missing, "get_audio_url", timeout_s=1)
    with pytest.raises(ResolveUnavailable, match="没有连上"):
        await resolver.resolve(_track())

    async def broken(name: str, args: dict[str, Any]) -> str:
        return "工具 get_audio_url 暂时不可用：调用超时。"

    resolver = UrlResolver(broken, "get_audio_url", timeout_s=1)
    with pytest.raises(ResolveUnavailable, match="没有响应"):
        await resolver.resolve(_track())

    async def rejected(name: str, args: dict[str, Any]) -> str:
        raise RuntimeError("no valid download url found or vip required")

    resolver = UrlResolver(rejected, "get_audio_url", timeout_s=1)
    with pytest.raises(ResolveUnavailable, match="没有响应") as caught:
        await resolver.resolve(_track())
    assert caught.value.__cause__ is not None


async def test_resolver_without_tool_does_not_call() -> None:
    """没配置取地址工具时，不发起调用。"""

    async def fail(name: str, args: dict[str, Any]) -> str:
        raise AssertionError("不应该调用")

    resolver = UrlResolver(fail, "", timeout_s=1)
    with pytest.raises(ResolveUnavailable, match="没有配置"):
        await resolver.resolve(_track())
