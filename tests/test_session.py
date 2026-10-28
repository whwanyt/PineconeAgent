"""会话门面：文本、打断和播放选择都转发到接上来的对象，不碰具体技能。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pinecone.core.bus import EventBus
from pinecone.core.state import StateMachine
from pinecone.core.task_manager import TaskManager
from pinecone.interaction import PlaylistSnapshot, Session


class _Dialog:
    def __init__(self) -> None:
        self.timer_remaining: float | None = 3.0
        self.texts: list[str] = []
        self.reasons: list[str] = []

    async def handle_text(self, text: str) -> None:
        self.texts.append(text)

    async def interrupt(self, reason: str = "user") -> int:
        self.reasons.append(reason)
        return 1


class _Playback:
    def __init__(self) -> None:
        self.snap = PlaylistSnapshot(
            tracks=("晴天", "稻香"), current=1, mode="顺序播放", paused=False
        )
        self.selected: list[int] = []

    def snapshot(self) -> PlaylistSnapshot:
        return self.snap

    async def select(self, index: int) -> None:
        self.selected.append(index)


def _session(tmp_path: Path) -> Session:
    ready = asyncio.Event()
    return Session(
        bus=EventBus(),
        state=StateMachine(EventBus()),
        tasks=TaskManager(EventBus()),
        ready=ready,
        assistant_name="小派",
        data_path=tmp_path,
        run=_no_run,
        stop=lambda: None,
    )


async def _no_run(*, stdin: bool = True) -> None:
    return None


async def test_commands_reach_dialog_and_playback(tmp_path: Path) -> None:
    """接上之后，文本和打断进对话控制器，选曲进播放控制，不经过另一边。"""
    session = _session(tmp_path)
    dialog = _Dialog()
    playback = _Playback()
    session.attach(dialog, playback)

    assert session.timer_remaining == 3.0
    assert session.playlist().tracks == ("晴天", "稻香")
    await session.submit_text("现在几点")
    await session.interrupt()
    await session.select_track(2)

    assert dialog.texts == ["现在几点"]
    assert dialog.reasons == ["user"]
    assert playback.selected == [2]


async def test_playback_is_optional(tmp_path: Path) -> None:
    """没有播放控制时列表为空，选曲什么都不做。"""
    session = _session(tmp_path)
    session.attach(_Dialog())
    assert session.playlist() == PlaylistSnapshot()
    await session.select_track(1)


async def test_submit_before_attach_fails(tmp_path: Path) -> None:
    """对话控制器还没接上时不能收文本。"""
    session = _session(tmp_path)
    with pytest.raises(RuntimeError, match="还没准备好"):
        await session.submit_text("你好")


async def test_stop_and_run_forward(tmp_path: Path) -> None:
    """退出和启动转给应用自己的回调，界面不直接碰应用对象。"""
    stopped = False
    seen: dict[str, bool] = {}

    async def run(*, stdin: bool = True) -> None:
        seen["stdin"] = stdin

    def stop() -> None:
        nonlocal stopped
        stopped = True

    ready = asyncio.Event()
    session = Session(
        bus=EventBus(),
        state=StateMachine(EventBus()),
        tasks=TaskManager(EventBus()),
        ready=ready,
        assistant_name="小派",
        data_path=tmp_path,
        run=run,
        stop=stop,
    )
    session.stop()
    await session.run(stdin=False)
    assert stopped and seen["stdin"] is False
