"""闹钟持久化（SQLite）。

闹钟的“真相来源”是这张表，调度器（APScheduler）只在内存里按表中数据排程。
这样重启时只需从表中重新加载，表结构也简单直观，可以直接用 sqlite3 命令查看。

表结构::

    alarms(
        id          INTEGER PRIMARY KEY,
        label       TEXT,      -- 提醒内容，如“起床”“开会”，可为空
        at          TEXT,      -- 一次性闹钟的触发时间（ISO 格式）；重复闹钟为 NULL
        hour        INTEGER,   -- 钟点
        minute      INTEGER,
        weekdays    TEXT,      -- 重复闹钟的星期，逗号分隔（0=周一）；一次性闹钟为 NULL
        created_at  TEXT
    )
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Alarm:
    """一个闹钟。"""

    id: int
    label: str
    hour: int
    minute: int
    at: datetime | None = None
    weekdays: list[int] | None = None

    @property
    def repeat(self) -> bool:
        return self.weekdays is not None


class AlarmStore:
    """闹钟表的增删查。

    线程安全：只在事件循环线程中调用（SQLite 操作很快，不需要放到线程池）。
    """

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alarms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL DEFAULT '',
                at TEXT,
                hour INTEGER NOT NULL,
                minute INTEGER NOT NULL,
                weekdays TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def add(
        self, *, label: str, hour: int, minute: int, at: datetime | None, weekdays: list[int] | None
    ) -> Alarm:
        """新增闹钟并返回（带自增 id）。"""
        cur = self._conn.execute(
            "INSERT INTO alarms(label, at, hour, minute, weekdays, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                label,
                at.isoformat() if at else None,
                hour,
                minute,
                ",".join(map(str, weekdays)) if weekdays is not None else None,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()
        return Alarm(
            id=int(cur.lastrowid or 0),
            label=label,
            hour=hour,
            minute=minute,
            at=at,
            weekdays=weekdays,
        )

    def all(self) -> list[Alarm]:
        """全部闹钟。"""
        rows = self._conn.execute(
            "SELECT id, label, at, hour, minute, weekdays FROM alarms ORDER BY id"
        ).fetchall()
        return [
            Alarm(
                id=row[0],
                label=row[1],
                at=datetime.fromisoformat(row[2]) if row[2] else None,
                hour=row[3],
                minute=row[4],
                weekdays=[int(x) for x in row[5].split(",") if x] if row[5] is not None else None,
            )
            for row in rows
        ]

    def delete(self, alarm_id: int) -> None:
        """删除闹钟。"""
        self._conn.execute("DELETE FROM alarms WHERE id = ?", (alarm_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
