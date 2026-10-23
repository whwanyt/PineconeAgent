"""全屏界面主体：组装各区域，把输入交给对话控制器，随后端一起启动和退出。"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.widgets import Input, RichLog, Static

from pinecone.tui.logs import TuiLogHandler

if TYPE_CHECKING:
    from pinecone.core.bus import EventBus
    from pinecone.dialog import DialogController

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class Backend(Protocol):
    """界面需要的后端接口，由 :class:`pinecone.app.App` 实现。"""

    bus: EventBus
    dialog: DialogController
    ready: asyncio.Event

    async def run(self, *, stdin: bool = True) -> None: ...

    def stop(self) -> None: ...


class PineconeTui(App[None]):
    """PineconeAgent 文本模式界面。

    后端在界面的 worker 里运行：界面退出键只请求后端停止，等后端关完资源再关界面，
    这样退出过程中的日志也能看到。
    """

    TITLE = "PineconeAgent"
    CSS = """
    #conversation {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    #log {
        height: 40%;
        border: round $secondary;
        padding: 0 1;
    }
    #status {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "退出", priority=True),
        Binding("ctrl+q", "quit", "退出", show=False, priority=True),
    ]

    def __init__(self, backend: Backend) -> None:
        super().__init__()
        self._backend = backend
        self._log_handler: TuiLogHandler | None = None
        # 发出去还没处理完的输入；持有引用，免得任务被垃圾回收
        self._sending: set[asyncio.Task[None]] = set()

    def compose(self) -> ComposeResult:
        yield RichLog(id="conversation", wrap=True, markup=False)
        yield RichLog(id="log", wrap=True, markup=False, max_lines=2000)
        yield Static("启动中……", id="status")
        yield Input(placeholder="启动中……", id="prompt", disabled=True)

    def on_mount(self) -> None:
        self.query_one("#conversation").border_title = "对话"
        self.query_one("#log").border_title = "日志"
        handler = TuiLogHandler(self._write_log, asyncio.get_running_loop())
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logging.getLogger().addHandler(handler)
        self._log_handler = handler
        self.run_worker(self._run_backend(), name="pinecone")
        self.run_worker(self._enable_input(), name="ready")

    def on_unmount(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)

    async def action_quit(self) -> None:
        self._backend.stop()

    async def _run_backend(self) -> None:
        try:
            await self._backend.run(stdin=False)
        finally:
            self.exit()

    async def _enable_input(self) -> None:
        await self._backend.ready.wait()
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = "输入一句话后回车（exit 退出）"
        prompt.disabled = False
        prompt.focus()
        self.query_one("#status", Static).update("就绪")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        if text in ("exit", "quit"):
            self._backend.stop()
            return
        self.query_one("#conversation", RichLog).write(f"你  {text}")
        # 不等这一轮处理完：思考或播报中接着输入，就像语音打断一样换成新的一句
        task = asyncio.create_task(self._send(text))
        self._sending.add(task)
        task.add_done_callback(self._sending.discard)

    async def _send(self, text: str) -> None:
        try:
            await self._backend.dialog.handle_text(text)
        except Exception:
            logger.exception("处理输入出错：%s", text)

    def _write_log(self, level: int, text: str) -> None:
        try:
            log = self.query_one("#log", RichLog)
        except NoMatches:  # 界面已经关闭，投递过来的最后几条日志在文件里
            return
        log.write(text)
