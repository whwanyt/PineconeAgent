"""自建歌单（SQLite）。

播放队列只活在内存里。用户说「收藏」「保存为跑步歌单」时，把曲目写到
``data/music/playlists.db``。在线歌只存歌曲信息，不存会过期的播放地址。

表结构::

    playlists(
        id          INTEGER PRIMARY KEY,
        name        TEXT UNIQUE,   -- 归一化后的歌单名，内置「我喜欢」
        created_at  TEXT
    )
    playlist_tracks(
        id           INTEGER PRIMARY KEY,  -- 自增顺序就是歌单顺序
        playlist_id  INTEGER,
        key          TEXT,                 -- 路径，或「来源:id」
        title        TEXT,
        artist       TEXT,
        uri          TEXT,                 -- 本地或电台地址；在线歌为空
        song         TEXT,                 -- 在线歌曲信息 JSON
        added_at     TEXT,
        UNIQUE(playlist_id, key)
    )

只在事件循环线程里调用。SQLite 操作很短，不放到线程池，也不响应取消。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .library import Track, search_key

FAVORITES = "我喜欢"
_FAVORITE_ALIASES = {"收藏", "收藏夹", "喜欢", "我喜欢的歌", "红心", "我喜欢"}


def canonical_name(name: str) -> str:
    """把口头的歌单名收成存储用的名字。

    去掉开头的「我的」和结尾的「歌单 / 列表 / 的」。收藏、红心这类说法都归到「我喜欢」。
    """
    text = name.strip()
    if text in _FAVORITE_ALIASES:
        return FAVORITES
    if text.startswith("我的"):
        text = text.removeprefix("我的")
    for suffix in ("歌单", "列表", "的"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    if text in _FAVORITE_ALIASES:
        return FAVORITES
    return text.strip()


@dataclass(frozen=True)
class Playlist:
    """一个歌单的名字和曲目数量。"""

    id: int
    name: str
    created_at: str
    size: int = 0


class PlaylistStore:
    """歌单表的增删查。"""

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS playlist_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                playlist_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                title TEXT NOT NULL,
                artist TEXT NOT NULL DEFAULT '',
                uri TEXT NOT NULL DEFAULT '',
                song TEXT NOT NULL DEFAULT '',
                added_at TEXT NOT NULL,
                UNIQUE(playlist_id, key)
            )
            """
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO playlists(name, created_at) VALUES (?, ?)",
            (FAVORITES, _now()),
        )
        self._conn.commit()

    def all(self) -> list[Playlist]:
        """全部歌单。「我喜欢」固定排在第一。"""
        rows = self._conn.execute(
            """
            SELECT p.id, p.name, p.created_at, COUNT(t.id)
            FROM playlists AS p
            LEFT JOIN playlist_tracks AS t ON t.playlist_id = p.id
            GROUP BY p.id
            ORDER BY CASE WHEN p.name = ? THEN 0 ELSE 1 END, p.id
            """,
            (FAVORITES,),
        ).fetchall()
        return [Playlist(id=row[0], name=row[1], created_at=row[2], size=row[3]) for row in rows]

    def find_matches(self, name: str) -> list[Playlist]:
        """先按归一化后的全名找，找不到再按互相包含找。"""
        canonical = canonical_name(name)
        if not canonical:
            return []
        playlists = self.all()
        exact = [item for item in playlists if item.name == canonical]
        if exact:
            return exact
        return [item for item in playlists if canonical in item.name or item.name in canonical]

    def find(self, name: str) -> Playlist | None:
        """精确或唯一包含时返回歌单。没有或不止一个时返回 None。"""
        found = self.find_matches(name)
        return found[0] if len(found) == 1 else None

    def create(self, name: str) -> Playlist:
        """按归一化名字新建。已经有同名歌单时直接返回那一个。"""
        canonical = canonical_name(name)
        if not canonical:
            raise ValueError("歌单名是空的")
        self._conn.execute(
            "INSERT OR IGNORE INTO playlists(name, created_at) VALUES (?, ?)",
            (canonical, _now()),
        )
        self._conn.commit()
        found = self.find(canonical)
        if found is None:
            raise ValueError(f"无法创建歌单{name}")
        return found

    def tracks(self, playlist_id: int) -> list[Track]:
        """歌单里的歌，按加入顺序。"""
        rows = self._conn.execute(
            """
            SELECT title, artist, uri, song
            FROM playlist_tracks
            WHERE playlist_id = ?
            ORDER BY id
            """,
            (playlist_id,),
        ).fetchall()
        return [
            Track(
                title=row[0],
                uri=row[2],
                search_key=search_key(f"{row[1]}{row[0]}"),
                artist=row[1],
                song=row[3],
            )
            for row in rows
        ]

    def add(self, playlist_id: int, tracks: list[Track]) -> int:
        """加入曲目。同一首歌已在歌单里时跳过，返回新加入的数量。"""
        added = 0
        for item in tracks:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO playlist_tracks(
                    playlist_id, key, title, artist, uri, song, added_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (playlist_id, item.key, item.title, item.artist, item.uri, item.song, _now()),
            )
            added += cur.rowcount
        self._conn.commit()
        return added

    def replace_tracks(self, playlist_id: int, tracks: list[Track]) -> int:
        """用这批歌替换歌单内容，顺序与传入一致。"""
        self._conn.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
        self._conn.commit()
        return self.add(playlist_id, tracks)

    def remove(self, playlist_id: int, key: str) -> bool:
        """按去重键删除一首。没有这首时返回 False。"""
        cur = self._conn.execute(
            "DELETE FROM playlist_tracks WHERE playlist_id = ? AND key = ?",
            (playlist_id, key),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def remove_at(self, playlist_id: int, position: int) -> Track | None:
        """删除从 1 数的那一首。位置不存在时返回 None。"""
        row = self._conn.execute(
            """
            SELECT id, title, artist, uri, song
            FROM playlist_tracks
            WHERE playlist_id = ?
            ORDER BY id
            LIMIT 1 OFFSET ?
            """,
            (playlist_id, position - 1),
        ).fetchone()
        if row is None:
            return None
        self._conn.execute("DELETE FROM playlist_tracks WHERE id = ?", (row[0],))
        self._conn.commit()
        return Track(title=row[1], uri=row[3], artist=row[2], song=row[4])

    def delete(self, playlist_id: int) -> None:
        """删除歌单及其曲目。「我喜欢」不能删。

        Raises:
            ValueError: 要删的是「我喜欢」，或歌单不存在。
        """
        row = self._conn.execute(
            "SELECT name FROM playlists WHERE id = ?", (playlist_id,)
        ).fetchone()
        if row is None:
            raise ValueError("没有这个歌单")
        if row[0] == FAVORITES:
            raise ValueError("我喜欢不能删除")
        self._conn.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
        self._conn.execute("DELETE FROM playlists WHERE id = ?", (playlist_id,))
        self._conn.commit()

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
