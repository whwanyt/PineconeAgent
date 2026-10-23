"""全屏文本界面测试（使用假后端，不需要声卡和模型）。"""

import asyncio
import logging

import pytest

pytest.importorskip("textual")

from textual.widgets import Input, RichLog  # noqa: E402

from pinecone.core.bus import EventBus  # noqa: E402
from pinecone.tui import PineconeTui  # noqa: E402


class FakeDialog:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def handle_text(self, text: str) -> None:
        self.texts.append(text)


class FakeBackend:
    """按 pinecone.app.App 的接口：run 启动并等到 stop，启动完成时 set ready。"""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.dialog = FakeDialog()
        self.ready = asyncio.Event()
        self.stdin: bool | None = None
        self.shut_down = False
        self._stop = asyncio.Event()

    async def run(self, *, stdin: bool = True) -> None:
        self.stdin = stdin
        self.bus.bind_loop(asyncio.get_running_loop())
        self.ready.set()
        await self._stop.wait()
        logging.getLogger("pinecone.app").info("正在退出……")
        self.shut_down = True

    def stop(self) -> None:
        self._stop.set()


def lines(app: PineconeTui, widget_id: str) -> list[str]:
    return [line.text for line in app.query_one(widget_id, RichLog).lines]


async def submit(pilot, text: str) -> None:
    pilot.app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await pilot.pause()


async def test_input_goes_to_dialog_and_logs_stay_apart() -> None:
    """输入框启动后可用；回车交给对话控制器；日志显示在日志区，不进对话区。"""
    backend = FakeBackend()
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.pause()
        assert backend.stdin is False
        assert not app.query_one("#prompt", Input).disabled

        logging.getLogger("pinecone.skills").warning("技能加载慢")
        await submit(pilot, "现在几点")
        assert backend.dialog.texts == ["现在几点"]
        assert app.query_one("#prompt", Input).value == ""
        assert any("现在几点" in line for line in lines(app, "#conversation"))
        assert any("技能加载慢" in line for line in lines(app, "#log"))
        assert not any("技能加载慢" in line for line in lines(app, "#conversation"))


async def test_quit_waits_for_backend_shutdown() -> None:
    """Ctrl+C 只请求后端停止，后端关完再退出界面；退出后不再接管日志。"""
    backend = FakeBackend()
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.press("ctrl+c")
        await asyncio.wait_for(backend._stop.wait(), 1)
    assert backend.shut_down
    root = logging.getLogger()
    assert not any(type(h).__name__ == "TuiLogHandler" for h in root.handlers)


async def test_exit_command_stops_backend() -> None:
    """输入 exit 与 Ctrl+C 一样退出，不交给对话控制器。"""
    backend = FakeBackend()
    app = PineconeTui(backend)
    async with app.run_test() as pilot:
        await backend.ready.wait()
        await pilot.pause()
        await submit(pilot, "exit")
        await asyncio.wait_for(backend._stop.wait(), 1)
    assert backend.dialog.texts == []
