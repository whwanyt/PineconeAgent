"""播放队列的模式、去重和增删，不启动播放器。"""

from __future__ import annotations

import random

import pytest

from pinecone.skills.builtin.music.library import Track, search_key
from pinecone.skills.builtin.music.playqueue import PlayMode, PlayQueue


def track(title: str, uri: str | None = None) -> Track:
    return Track(title=title, uri=uri or title, search_key=search_key(title))


def titles(queue: PlayQueue) -> list[str]:
    return [item.title for item in queue.tracks]


def test_track_key_label_and_song_info() -> None:
    """本地歌用路径去重；在线歌用来源和 id，播报时带上歌手。"""
    local = Track(title="晴天", uri="/music/晴天.mp3", artist="周杰伦")
    online = Track(
        title="九万字",
        uri="",
        artist="黄诗扶",
        song='{"id":"9","source":"netease","name":"九万字"}',
    )
    assert local.key == "/music/晴天.mp3"
    assert local.label == "周杰伦的晴天"
    assert online.online
    assert online.key == "netease:9"
    assert online.label == "黄诗扶的九万字"
    assert online.song_info()["source"] == "netease"
    assert Track(title="坏数据", uri="", song="[").song_info() == {}


def test_replace_dedups_and_respects_limit() -> None:
    """整表替换时丢掉重复项和超出上限的歌，并从第一首开始。"""
    queue = PlayQueue(limit=2)
    kept = queue.replace([track("甲"), track("甲"), track("乙"), track("丙")])
    assert [item.title for item in kept] == ["甲", "乙"]
    assert queue.position == 1
    assert queue.current is not None and queue.current.title == "甲"


def test_append_and_insert_next() -> None:
    """追加跳过已有的歌；插播会把已经在后面的歌挪到下一首，当前这首不动。"""
    queue = PlayQueue()
    queue.replace([track("甲"), track("乙"), track("丙")])

    assert queue.append([track("甲"), track("丁")])[0].title == "丁"
    assert titles(queue) == ["甲", "乙", "丙", "丁"]

    moved = queue.insert_next([track("丙"), track("戊")])
    assert [item.title for item in moved] == ["丙", "戊"]
    assert titles(queue) == ["甲", "丙", "戊", "乙", "丁"]
    assert queue.current is not None and queue.current.title == "甲"

    assert queue.insert_next([track("甲")]) == []
    assert queue.position == 1


def test_insert_next_moves_earlier_track_without_losing_current() -> None:
    """把前面的歌插到当前曲后面时，当前曲还是原来那首。"""
    queue = PlayQueue()
    queue.replace([track("甲"), track("乙"), track("丙")])
    queue.jump(2)
    queue.insert_next([track("甲")])
    assert titles(queue) == ["乙", "甲", "丙"]
    assert queue.current is not None and queue.current.title == "乙"


def test_append_to_empty_starts_at_first() -> None:
    """空队列上追加就是开始播放。"""
    queue = PlayQueue()
    queue.append([track("甲")])
    assert queue.position == 1


def test_remove_before_current_and_current_itself() -> None:
    """删当前曲前面的歌时位置前移；删当前曲则接到后面一首；删掉最后的当前曲后当前为空。"""
    queue = PlayQueue()
    queue.replace([track("甲"), track("乙"), track("丙")])
    queue.jump(3)

    removed, was_current = queue.remove(1)
    assert removed.title == "甲" and not was_current
    assert queue.current is not None and queue.current.title == "丙"
    assert queue.position == 2

    removed, was_current = queue.remove(2)
    assert removed.title == "丙" and was_current
    assert queue.current is None
    assert titles(queue) == ["乙"]


def test_remove_current_middle_advances() -> None:
    """删掉中间正在放的歌后，原来的下一首成为当前曲。"""
    queue = PlayQueue()
    queue.replace([track("甲"), track("乙"), track("丙")])
    queue.jump(2)
    removed, was_current = queue.remove(2)
    assert removed.title == "乙" and was_current
    assert queue.current is not None and queue.current.title == "丙"


def test_remove_out_of_range() -> None:
    """越界删除会说明现在有几首。"""
    queue = PlayQueue()
    queue.replace([track("甲")])
    with pytest.raises(IndexError, match="只有1首"):
        queue.remove(2)


def test_clear_others_jump_and_lookup() -> None:
    """清空其余、跳转，以及同名歌要能列出全部位置。"""
    queue = PlayQueue()
    queue.replace([track("晴天", "a"), track("稻香"), track("晴天", "b")])
    queue.jump(2)
    assert queue.clear_others() == 2
    assert titles(queue) == ["稻香"]

    queue.replace([track("晴天", "a"), track("稻香"), track("晴天", "b")])
    assert queue.matches("晴天") == [1, 3]
    assert queue.find("晴天") is None
    assert queue.find("稻香") == 2
    assert queue.upcoming(1)[0].title == "稻香"
    assert queue.peek_next() is not None and queue.peek_next().title == "稻香"


@pytest.mark.parametrize(
    ("mode", "auto", "expected"),
    [
        (PlayMode.ORDER, True, ["乙", "丙", None]),
        (PlayMode.ORDER, False, ["乙", "丙", None]),
        (PlayMode.LOOP, True, ["乙", "丙", "甲"]),
        (PlayMode.LOOP, False, ["乙", "丙", "甲"]),
        (PlayMode.SINGLE, True, ["甲", "甲", "甲"]),
        (PlayMode.SINGLE, False, ["乙", "丙", "甲"]),
    ],
)
def test_next_by_mode(mode: PlayMode, auto: bool, expected: list[str | None]) -> None:
    """顺序、列表循环、单曲循环的自动换歌和手动下一首。"""
    queue = PlayQueue(mode)
    queue.replace([track("甲"), track("乙"), track("丙")])
    got: list[str | None] = []
    for _ in expected:
        nxt = queue.next(auto=auto)
        got.append(None if nxt is None else nxt.title)
    assert got == expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (PlayMode.ORDER, [None, "甲"]),
        (PlayMode.SHUFFLE, [None, "甲"]),
        (PlayMode.LOOP, ["丙", "甲"]),
        (PlayMode.SINGLE, ["丙", "甲"]),
    ],
)
def test_previous_by_mode(mode: PlayMode, expected: list[str | None]) -> None:
    """第一首再往前：顺序和随机停住，列表循环和单曲循环回到最后。"""
    queue = PlayQueue()
    queue.replace([track("甲"), track("乙"), track("丙")])
    queue.set_mode(mode)
    first = queue.previous()
    queue.jump(2)
    second = queue.previous()
    assert [None if item is None else item.title for item in (first, second)] == expected


def test_shuffle_keeps_current_first_and_starts_new_round_elsewhere() -> None:
    """随机模式不丢歌，当前曲留在第一首，新一轮不以刚放完的那首开头。"""
    queue = PlayQueue(rng=random.Random(0))
    queue.replace([track(name) for name in "甲乙丙丁"])
    queue.set_mode(PlayMode.SHUFFLE)

    assert queue.current is not None and queue.current.title == "甲"
    assert set(titles(queue)) == set("甲乙丙丁")
    seen = [queue.current.title]
    for _ in range(3):
        nxt = queue.next(auto=False)
        assert nxt is not None
        seen.append(nxt.title)
    assert seen[0] == "甲"
    assert set(seen) == set("甲乙丙丁")

    finished = queue.current.title if queue.current is not None else ""
    nxt = queue.next(auto=True)
    assert nxt is not None
    assert nxt.title != finished
    assert set(titles(queue)) == set("甲乙丙丁")
    assert queue.position == 1


def test_peek_next_single_is_current_and_shuffle_end_is_unknown() -> None:
    """单曲循环预取的就是当前曲；随机一轮的最后一首还不知道下一轮是谁。"""
    single = PlayQueue(PlayMode.SINGLE)
    single.replace([track("甲"), track("乙")])
    assert single.peek_next() is not None and single.peek_next().title == "甲"

    shuffle = PlayQueue(PlayMode.SHUFFLE, rng=random.Random(1))
    shuffle.replace([track("甲")])
    shuffle.jump(1)
    assert shuffle.peek_next() is None
