"""自建歌单的名字归一化、去重和重启后仍在。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pinecone.skills.builtin.music.library import Track
from pinecone.skills.builtin.music.store import FAVORITES, PlaylistStore, canonical_name


def track(title: str, *, uri: str | None = None, song: str = "", artist: str = "") -> Track:
    return Track(title=title, uri=title if uri is None else uri, artist=artist, song=song)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("我喜欢", FAVORITES),
        ("收藏", FAVORITES),
        ("收藏夹", FAVORITES),
        ("喜欢", FAVORITES),
        ("我喜欢的歌", FAVORITES),
        ("红心", FAVORITES),
        ("我的喜欢", FAVORITES),
        ("跑步歌单", "跑步"),
        ("我的跑步列表", "跑步"),
        ("睡前的", "睡前"),
    ],
)
def test_canonical_name(raw: str, expected: str) -> None:
    """口头说法收成存储用的歌单名。"""
    assert canonical_name(raw) == expected


def test_favorites_exist_and_cannot_be_deleted(tmp_path: Path) -> None:
    """新建库就有「我喜欢」，而且删不掉。"""
    store = PlaylistStore(tmp_path / "playlists.db")
    playlists = store.all()
    assert [item.name for item in playlists] == [FAVORITES]
    with pytest.raises(ValueError, match="不能删除"):
        store.delete(playlists[0].id)
    store.close()


def test_add_is_ordered_and_deduped(tmp_path: Path) -> None:
    """重复加入不计数，顺序保持第一次加入的顺序。"""
    store = PlaylistStore(tmp_path / "playlists.db")
    favorites = store.find("收藏")
    assert favorites is not None
    first = track("晴天", uri="/a.mp3", artist="周杰伦")
    assert store.add(favorites.id, [first, first, track("稻香", uri="/b.mp3")]) == 2
    assert store.add(favorites.id, [first]) == 0
    assert [item.title for item in store.tracks(favorites.id)] == ["晴天", "稻香"]
    assert store.all()[0].size == 2
    store.close()


def test_remove_at_and_replace(tmp_path: Path) -> None:
    """按位置删除，以及保存播放列表时整表换掉。"""
    store = PlaylistStore(tmp_path / "playlists.db")
    playlist = store.create("跑步歌单")
    store.add(playlist.id, [track("甲", uri="1"), track("乙", uri="2"), track("丙", uri="3")])
    removed = store.remove_at(playlist.id, 2)
    assert removed is not None and removed.title == "乙"
    assert [item.title for item in store.tracks(playlist.id)] == ["甲", "丙"]
    assert store.remove_at(playlist.id, 5) is None

    assert store.replace_tracks(playlist.id, [track("丁", uri="4")]) == 1
    assert [item.title for item in store.tracks(playlist.id)] == ["丁"]
    store.close()


def test_delete_playlist_removes_tracks(tmp_path: Path) -> None:
    """删歌单时曲目一起删掉，不能再按旧 id 找回来。"""
    store = PlaylistStore(tmp_path / "playlists.db")
    playlist = store.create("睡前")
    store.add(playlist.id, [track("甲", uri="1")])
    store.delete(playlist.id)
    assert store.find("睡前") is None
    assert store.tracks(playlist.id) == []
    store.close()


def test_reopen_keeps_online_song(tmp_path: Path) -> None:
    """关掉再打开，在线歌的歌曲信息还在，地址仍然是空的。"""
    path = tmp_path / "playlists.db"
    song = '{"id":"9","source":"netease","name":"九万字"}'
    store = PlaylistStore(path)
    favorites = store.create("我喜欢")
    store.add(favorites.id, [track("九万字", uri="", song=song, artist="黄诗扶")])
    store.close()

    again = PlaylistStore(path)
    favorites = again.find("红心")
    assert favorites is not None
    kept = again.tracks(favorites.id)
    assert kept[0].title == "九万字"
    assert kept[0].artist == "黄诗扶"
    assert kept[0].uri == ""
    assert kept[0].song == song
    assert kept[0].online
    again.close()


def test_ambiguous_name_is_not_chosen(tmp_path: Path) -> None:
    """名字互相包含且不止一个时，不替用户挑。"""
    store = PlaylistStore(tmp_path / "playlists.db")
    store.create("跑步")
    store.create("跑步机")
    assert store.find("跑") is None
    assert [item.name for item in store.find_matches("跑")] == ["跑步", "跑步机"]
    assert store.find("跑步机") is not None
    store.close()
