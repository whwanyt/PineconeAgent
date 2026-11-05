"""全屏界面测试（使用假会话门面，不需要声卡和模型）。"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, RichLog  # noqa: E402

from pinecone.core.bus import EventBus  # noqa: E402
from pinecone.core.events import ChoicesShown, PlaylistUpdated, Spoken, Transcribed  # noqa: E402
from pinecone.core.state import AssistantState as S  # noqa: E402
from pinecone.core.state import StateMachine  # noqa: E402
from pinecone.core.task_manager import TaskKind, TaskManager  # noqa: E402
from pinecone.interaction import PlaylistSnapshot  # noqa: E402
from pinecone.tui import PineconeTui  # noqa: E402
from pinecone.tui.conversation import ConversationView  # noqa: E402
from pinecone.tui.logs import LogView  # noqa: E402
from pinecone.tui.native import native_output_to_log  # noqa: E402
from pinecone.tui.playlist import PlaylistView, TrackRow  # noqa: E402
from pinecone.tui.prompt import PromptInput  # noqa: E402
from pinecone.tui.status import StatusBar  # noqa: E402


class FakeSession:
    """按会话门面的接口：run 启动并等到 stop，启动完成时 set ready。"""

    def __init__(self, data_dir: Path) -> None:
        self.assistant_name = "小派"
        self.data_path = data_dir
        self.bus = EventBus()
        self.state = StateMachine(self.bus)
        self.tasks = TaskManager(self.bus)
        self.ready = asyncio.Event()
        self.texts: list[str] = []
        self.timer_remaining: float | None = None
        self.interrupted = 0
        self.selected: list[int] = []
        self.queue = PlaylistSnapshot()
        self.shut_down = False
        self._stop = asyncio.Event()

    async def run(self) -> None:
        self.bus.bind_loop(asyncio.get_running_loop())
        self.ready.set()
        await self._stop.wait()
        logging.getLogger("pinecone.app").info("正在退出……")
        self.shut_down = True

    def stop(self) -> None:
        self._stop.set()

    async def submit_text(self, text: str) -> None:
        self.texts.append(text)
        await self.bus.publish(Transcribed(text=text))

    async def interrupt(self) -> None:
        self.interrupted += 1

    def playlist(self) -> PlaylistSnapshot:
        return self.queue

    async def select_track(self, index: int) -> None:
        self.selected.append(index)


def lines(app: PineconeTui, widget_id: str) -> list[str]:
    if widget_id == "#conversation":
        return [f"{m.speaker}|{m.body}" for m in app.query_one(ConversationView).messages]
    return [line.text for line in app.query_one(widget_id, RichLog).lines]


async def submit(pilot, text: str) -> None:
    pilot.app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await pilot.pause()


async def test_input_goes_to_dialog_and_logs_stay_apart(tmp_path: Path) -> None:
    """输入框启动后可用；回车交给会话门面；日志显示在日志区，不进对话区。"""
    backend = FakeSession(tmp_path)
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.pause()
        assert not app.query_one("#prompt", Input).disabled

        logging.getLogger("pinecone.skills").warning("技能加载慢")
        await submit(pilot, "现在几点")
        assert backend.texts == ["现在几点"]
        assert app.query_one("#prompt", Input).value == ""
        assert any("现在几点" in line for line in lines(app, "#conversation"))
        assert any("技能加载慢" in line for line in lines(app, "#log"))
        assert not any("技能加载慢" in line for line in lines(app, "#conversation"))


async def test_quit_waits_for_backend_shutdown(tmp_path: Path) -> None:
    """Ctrl+C 只请求后端停止，后端关完再退出界面；退出后不再接管日志。"""
    backend = FakeSession(tmp_path)
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.press("ctrl+c")
        await asyncio.wait_for(backend._stop.wait(), 1)
    assert backend.shut_down
    root = logging.getLogger()
    assert not any(type(h).__name__ == "TuiLogHandler" for h in root.handlers)


async def test_exit_command_stops_backend(tmp_path: Path) -> None:
    """输入 exit 与 Ctrl+C 一样退出，不交给会话门面。"""
    backend = FakeSession(tmp_path)
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.pause()
        await submit(pilot, "exit")
        await asyncio.wait_for(backend._stop.wait(), 1)
    assert backend.texts == []


async def test_conversation_merges_one_utterance(tmp_path: Path) -> None:
    """同一次播报的句子拼成一条；新的播报、候选列表各占一条；用户的话以识别结果为准。"""
    backend = FakeSession(tmp_path)
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        bus = backend.bus
        await bus.publish(Transcribed(text="放歌"))
        await bus.publish(Spoken(text="找到两首。", utterance=1))
        await bus.publish(Spoken(text="要听哪一首？", utterance=1))
        await bus.publish(ChoicesShown(lines="1. 晴天\n2. 稻香"))
        await bus.publish(Spoken(text="闹钟响了。", utterance=2))
        await pilot.pause()
        assert lines(app, "#conversation") == [
            "你|放歌",
            "小派|找到两首。要听哪一首？",
            "|1. 晴天\n2. 稻香",
            "小派|闹钟响了。",
        ]


async def test_log_filter_and_toggle(tmp_path: Path) -> None:
    """F3 只看警告和错误、再按恢复全部，之前的日志不丢；F2 隐藏和显示日志区。"""
    backend = FakeSession(tmp_path)
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        log = logging.getLogger("pinecone.test")
        log.setLevel(logging.INFO)
        log.info("普通信息")
        log.warning("有点问题")
        await pilot.pause()
        assert any("普通信息" in line for line in lines(app, "#log"))

        # 光标在输入框里时快捷键也要生效
        await pilot.press("f3")
        assert any("有点问题" in line for line in lines(app, "#log"))
        assert not any("普通信息" in line for line in lines(app, "#log"))
        await pilot.press("f3")
        assert any("普通信息" in line for line in lines(app, "#log"))

        view = app.query_one(LogView)
        await pilot.press("f2")
        assert not view.display
        await pilot.press("f2")
        assert view.display


async def test_status_bar_shows_state_countdown_and_tasks(tmp_path: Path) -> None:
    """状态栏显示当前状态、追问倒计时和技能任务，不列对话流程自己的任务。"""
    backend = FakeSession(tmp_path)
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.pause()
        bar = app.query_one(StatusBar)

        def text() -> str:
            bar.refresh_status()
            return str(bar.content)

        assert "空闲" in text()
        await backend.state.transition(S.LISTENING)
        await backend.state.transition(S.THINKING)
        backend.tasks.spawn(asyncio.sleep(10), name="对话", source="dialog")
        backend.tasks.spawn(
            asyncio.sleep(10), name="音乐", kind=TaskKind.BACKGROUND, source="skill:music"
        )
        assert "思考中" in text() and "任务：音乐" in text() and "对话" not in text()

        await backend.state.transition(S.FOLLOW_UP)
        backend.timer_remaining = 7.6
        assert "等待追问" in text() and "8s" in text()
        await backend.tasks.shutdown()


async def test_history_survives_restart(tmp_path: Path) -> None:
    """↑↓ 翻历史，翻回来恢复正在输入的内容；历史存在数据目录，重启后还在。"""
    backend = FakeSession(tmp_path)
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.pause()
        await submit(pilot, "现在几点")
        await submit(pilot, "明天天气")
        await submit(pilot, "明天天气")
        prompt = app.query_one(PromptInput)
        prompt.value = "打到一半"
        await pilot.press("up")
        assert prompt.value == "明天天气"
        await pilot.press("up", "up")
        assert prompt.value == "现在几点"
        await pilot.press("down", "down")
        assert prompt.value == "打到一半"

    backend = FakeSession(tmp_path)
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.pause()
        await pilot.press("up")
        assert app.query_one(PromptInput).value == "明天天气"


async def test_playlist_toggle_select_and_follow(tmp_path: Path) -> None:
    """F4 隐藏播放列表；回车切到高亮的那首；没焦点时高亮跟着当前曲，选歌时不拽走光标。"""
    backend = FakeSession(tmp_path)
    backend.queue = PlaylistSnapshot(
        tracks=("晴天", "稻香"), current=1, mode="顺序播放", paused=False
    )
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.pause()
        view = app.query_one(PlaylistView)
        rows = list(view.query(TrackRow))
        assert [row.track_title for row in rows] == ["晴天", "稻香"]
        assert rows[0].current and view.index == 0

        await pilot.press("f4")
        assert not view.display and not view.can_focus
        await pilot.press("f4")
        assert view.display and view.can_focus

        await pilot.press("tab")
        assert app.focused is view
        await pilot.press("down", "enter")
        await pilot.pause()
        assert backend.selected == [2]

        app.query_one(PromptInput).focus()
        await backend.bus.publish(
            PlaylistUpdated(tracks=("晴天", "稻香"), current=2, mode="顺序播放", paused=False)
        )
        await pilot.pause()
        assert view.index == 1 and list(view.query(TrackRow))[1].current

        view.focus()
        await pilot.press("up")
        await backend.bus.publish(
            PlaylistUpdated(
                tracks=("晴天", "稻香", "十年"), current=2, mode="顺序播放", paused=False
            )
        )
        await pilot.pause()
        assert view.index == 0
        assert [row.track_title for row in view.query(TrackRow)] == ["晴天", "稻香", "十年"]

        app.query_one(PromptInput).focus()
        await submit(pilot, "现在几点")
        await pilot.press("up")
        assert app.query_one(PromptInput).value == "现在几点"


async def test_escape_interrupts_and_ctrl_l_clears(tmp_path: Path) -> None:
    """Esc 打断这一轮；Ctrl+L 清空对话区和日志区。"""
    backend = FakeSession(tmp_path)
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.pause()
        await submit(pilot, "讲个长故事")
        await pilot.press("escape")
        assert backend.interrupted == 1

        await pilot.press("ctrl+l")
        await pilot.pause()
        assert lines(app, "#conversation") == [] and lines(app, "#log") == []


def test_native_output_goes_to_log(caplog: pytest.LogCaptureFixture) -> None:
    """C 库直接写 fd 2 的字进日志；界面拿到的是另一份终端；退出后 1、2 号还原。"""
    before = os.fstat(2).st_ino, sys.__stderr__
    with caplog.at_level(logging.INFO, logger="native"), native_output_to_log():
        assert sys.__stderr__.fileno() != 2
        os.write(2, b"Unknown token: shei2\n")
        os.write(1, b"from stdout\n")
    assert os.fstat(2).st_ino == before[0] and sys.__stderr__ is before[1]
    assert [r.getMessage() for r in caplog.records if r.name == "native"] == [
        "Unknown token: shei2",
        "from stdout",
    ]


@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"), reason="Python 3.12 才有 eager 任务"
)
async def test_backend_tasks_are_not_eager(tmp_path: Path) -> None:
    """run_async 会把事件循环换成 eager 任务；界面换回默认，很快结束的任务照样从列表里清掉。"""
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)  # 和 Textual 的 run_async 一样
    try:
        backend = FakeSession(tmp_path)
        app = PineconeTui(backend)
        async with app.run_test() as pilot:
            await backend.ready.wait()
            await pilot.pause()
            assert loop.get_task_factory() is None

            async def quick() -> str:
                return "现在是上午11点"

            rt = await backend.tasks.run(quick(), name="查询时间和日期", source="skill:clock")
            assert rt.result == "现在是上午11点" and backend.tasks.list() == []
    finally:
        loop.set_task_factory(None)
