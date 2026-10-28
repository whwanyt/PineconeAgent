"""当前播放队列。

mpv 只负责正在听的这一首，队列、当前位置和播放模式都在这里。
音乐技能在事件循环线程里改队列，再按 :meth:`PlayQueue.current` 去加载播放器。
本模块不做网络请求，也不碰 mpv，因此可以脱离播放器做完整测试。

四种模式：

- 顺序播放：放到最后一首结束；
- 列表循环：最后一首之后回到第一首；
- 单曲循环：自动重放当前这首，手动「下一首」仍按列表循环切歌；
- 随机播放：当前曲留在第一位，其余打乱。一轮里每首一次，放完再洗牌，
  刚放完的那首不排在新一轮的第一首。
"""

from __future__ import annotations

import random
from enum import StrEnum

from .library import Track, search_key


class PlayMode(StrEnum):
    """播放模式。取值就是用户说的那句话，也是配置里写的字。"""

    ORDER = "顺序播放"
    LOOP = "列表循环"
    SINGLE = "单曲循环"
    SHUFFLE = "随机播放"


class PlayQueue:
    """内存中的播放队列。

    所有方法都只在事件循环线程调用，不阻塞，也不响应取消。
    同一份队列不要交给别的线程。
    """

    def __init__(
        self,
        mode: PlayMode = PlayMode.ORDER,
        *,
        limit: int = 100,
        rng: random.Random | None = None,
    ) -> None:
        self.mode = mode
        self._limit = max(1, limit)
        self._rng = rng or random.Random()
        self._tracks: list[Track] = []
        self._index = -1

    def __len__(self) -> int:
        return len(self._tracks)

    @property
    def tracks(self) -> list[Track]:
        """当前顺序的副本。调用方改这个列表不会影响队列。"""
        return list(self._tracks)

    @property
    def current(self) -> Track | None:
        """正在播放的曲目。队列空着或已经放到头时为 None。"""
        if not 0 <= self._index < len(self._tracks):
            return None
        return self._tracks[self._index]

    @property
    def position(self) -> int:
        """当前位置，从 1 数。没有当前曲时为 0。"""
        return 0 if self.current is None else self._index + 1

    def upcoming(self, n: int) -> list[Track]:
        """当前曲之后的最多 ``n`` 首，不含当前曲。"""
        if self._index < 0 or n <= 0:
            return []
        return self._tracks[self._index + 1 : self._index + 1 + n]

    def peek_next(self) -> Track | None:
        """自动换歌时的下一首，不移动当前位置。

        单曲循环就是当前这首。随机播放走到这一轮末尾时，下一轮的顺序还没洗出来，返回 None。
        """
        if self.current is None:
            return None
        if self.mode is PlayMode.SINGLE:
            return self.current
        if self._index + 1 < len(self._tracks):
            return self._tracks[self._index + 1]
        if self.mode is PlayMode.LOOP and self._tracks:
            return self._tracks[0]
        return None

    def matches(self, query: str) -> list[int]:
        """按歌名或搜索词找出所有位置（从 1 数）。"""
        key = search_key(query)
        if not key:
            return []
        found: list[int] = []
        for index, track in enumerate(self._tracks, 1):
            title_key = search_key(track.title)
            blob = track.search_key or title_key
            if key in (title_key, blob) or key in blob:
                found.append(index)
        return found

    def find(self, query: str) -> int | None:
        """只命中一首时返回位置，没有或多首都返回 None。"""
        found = self.matches(query)
        return found[0] if len(found) == 1 else None

    def replace(self, tracks: list[Track]) -> list[Track]:
        """换成这一批歌，并从第一首开始。

        按 :attr:`Track.key` 去重，超出上限的丢掉。已经处于随机模式时，
        第一首保持不动，其余打乱。返回实际留下的曲目（随机后的顺序）。
        """
        admitted = self._dedup(tracks, limit=self._limit)
        self._tracks = admitted
        self._index = 0 if admitted else -1
        if self.mode is PlayMode.SHUFFLE and len(self._tracks) > 1:
            self._shuffle_rest()
        return self.tracks

    def append(self, tracks: list[Track]) -> list[Track]:
        """加到末尾。队列是空的时等同于 :meth:`replace`。已存在的歌跳过。"""
        if not self._tracks:
            return self.replace(tracks)
        added: list[Track] = []
        seen = {track.key for track in self._tracks}
        for track in tracks:
            if len(self._tracks) >= self._limit:
                break
            if track.key in seen:
                continue
            seen.add(track.key)
            self._tracks.append(track)
            added.append(track)
        return added

    def insert_next(self, tracks: list[Track]) -> list[Track]:
        """插到当前曲的后面，并保持传入顺序。

        已经在队列里的歌会挪到这个位置。当前正在放的这一首不动。
        队列是空的时等同于 :meth:`replace`。
        """
        if not self._tracks:
            return self.replace(tracks)
        added: list[Track] = []
        insert_at = self._index + 1
        for track in tracks:
            existing = next(
                (i for i, item in enumerate(self._tracks) if item.key == track.key), None
            )
            if existing is None:
                if len(self._tracks) >= self._limit:
                    continue
                self._tracks.insert(insert_at, track)
                added.append(track)
                insert_at += 1
                continue
            if existing == self._index:
                continue
            item = self._tracks.pop(existing)
            if existing < self._index:
                self._index -= 1
            if existing < insert_at:
                insert_at -= 1
            self._tracks.insert(insert_at, item)
            added.append(item)
            insert_at += 1
        return added

    def remove(self, position: int) -> tuple[Track, bool]:
        """删掉从 1 数的那一首。

        Returns:
            被删的曲目，以及删的是不是当前曲。删掉当前曲且后面还有歌时，当前位置改为后面那首。
            删掉的是最后一首当前曲时，当前曲变为空，前面的歌还留在队列里。

        Raises:
            IndexError: 位置超出队列。
        """
        index = position - 1
        if not 0 <= index < len(self._tracks):
            raise IndexError(f"播放列表只有{len(self._tracks)}首")
        was_current = index == self._index
        track = self._tracks.pop(index)
        if not self._tracks:
            self._index = -1
        elif index < self._index:
            self._index -= 1
        elif was_current and self._index >= len(self._tracks):
            self._index = -1
        return track, was_current

    def clear_others(self) -> int:
        """只留下当前曲，返回删掉的数量。没有当前曲时清空。"""
        current = self.current
        if current is None:
            removed = len(self._tracks)
            self.clear()
            return removed
        removed = len(self._tracks) - 1
        self._tracks = [current]
        self._index = 0
        return removed

    def clear(self) -> None:
        """清空队列和当前位置。"""
        self._tracks = []
        self._index = -1

    def jump(self, position: int) -> Track:
        """跳到从 1 数的那一首。

        Raises:
            IndexError: 位置超出队列。
        """
        index = position - 1
        if not 0 <= index < len(self._tracks):
            raise IndexError(f"播放列表只有{len(self._tracks)}首")
        self._index = index
        return self._tracks[index]

    def next(self, *, auto: bool) -> Track | None:
        """切到下一首。

        Args:
            auto: 是否是播完后的自动换歌。单曲循环只有自动换歌才重放当前曲。

        Returns:
            新的当前曲。顺序播放已经在最后一首时返回 None，当前位置不动。
        """
        if self.current is None:
            return None
        if self.mode is PlayMode.SINGLE and auto:
            return self.current
        if self.mode is PlayMode.SHUFFLE:
            return self._shuffle_next()
        if self._index + 1 < len(self._tracks):
            self._index += 1
            return self.current
        if self.mode in (PlayMode.LOOP, PlayMode.SINGLE):
            self._index = 0
            return self.current
        return None

    def previous(self) -> Track | None:
        """切到上一首。顺序播放和随机播放在第一首时返回 None。"""
        if self.current is None:
            return None
        if self._index > 0:
            self._index -= 1
            return self.current
        if self.mode in (PlayMode.LOOP, PlayMode.SINGLE) and self._tracks:
            self._index = len(self._tracks) - 1
            return self.current
        return None

    def set_mode(self, mode: PlayMode) -> None:
        """切换模式。切到随机时，当前曲放到第一首，其余打乱。"""
        self.mode = mode
        if mode is PlayMode.SHUFFLE and self.current is not None and len(self._tracks) > 1:
            self._shuffle_rest()

    def _shuffle_rest(self) -> None:
        """当前曲留在第一位，后面的顺序打乱。"""
        current = self._tracks.pop(self._index)
        self._rng.shuffle(self._tracks)
        self._tracks.insert(0, current)
        self._index = 0

    def _shuffle_next(self) -> Track | None:
        """随机模式的下一首。这一轮放完后重新洗牌，刚放完的不排第一。"""
        if self._index + 1 < len(self._tracks):
            self._index += 1
            return self.current
        if len(self._tracks) == 1:
            return self.current
        finished = self._tracks[self._index]
        order = list(self._tracks)
        self._rng.shuffle(order)
        if order[0].key == finished.key:
            order.append(order.pop(0))
        self._tracks = order
        self._index = 0
        return self.current

    @staticmethod
    def _dedup(tracks: list[Track], *, limit: int) -> list[Track]:
        seen: set[str] = set()
        admitted: list[Track] = []
        for track in tracks:
            if track.key in seen:
                continue
            if len(admitted) >= limit:
                break
            seen.add(track.key)
            admitted.append(track)
        return admitted
