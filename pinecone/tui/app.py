"""全屏界面主体：组装各区域，把输入交给对话控制器，随后端一起启动和退出。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.widgets import Footer, Input

from pinecone.core.events import ChoicesShown, Spoken, Transcribed
from pinecone.tui.conversation import ConversationView
from pinecone.tui.logs import LogView, TuiLogHandler
from pinecone.tui.prompt import PromptInput
from pinecone.tui.status import StatusBar

if TYPE_CHECKING:
    from pinecone.core.bus import EventBus
    from pinecone.core.config import AppConfig
    from pinecone.core.state import StateMachine
    from pinecone.core.task_manager import TaskManager
    from pinecone.dialog import DialogController

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class Backend(Protocol):
    """界面需要的后端接口，由 :class:`pinecone.app.App` 实现。"""

    config: AppConfig
    bus: EventBus
    state: StateMachine
    tasks: TaskManager
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
    #log.hidden {
        display: none;
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
        Binding("escape", "interrupt", "打断", priority=True),
        Binding("ctrl+l", "clear", "清屏", priority=True),
        Binding("f2", "toggle_log", "日志"),
        Binding("f3", "warnings_only", "只看警告"),
    ]

    def __init__(self, backend: Backend) -> None:
        super().__init__()
        self._backend = backend
        self._log_handler: TuiLogHandler | None = None
        # 发出去还没处理完的输入；持有引用，免得任务被垃圾回收
        self._sending: set[asyncio.Task[None]] = set()
        self._unsubscribe: list[Callable[[], None]] = []

    def compose(self) -> ComposeResult:
        yield ConversationView(self._backend.config.name, id="conversation")
        yield LogView(id="log")
        yield StatusBar(id="status")
        yield PromptInput(
            self._backend.config.data_path / "tui_history.txt",
            placeholder="启动中……",
            id="prompt",
            disabled=True,
        )
        yield Footer()

    def on_mount(self) -> None:
        # Textual 在 Python 3.12+ 上把整个事件循环换成 eager 任务：create_task 当场就开始跑。
        # 后端和界面共用这个循环，TaskManager 等处都按“create_task 之后才开始跑”写的，
        # 很快结束的任务会来不及登记、永远留在任务列表里。换回默认，和树莓派（3.11）上一致
        asyncio.get_running_loop().set_task_factory(None)
        self.query_one("#conversation").border_title = "对话"
        self._update_log_title()
        handler = TuiLogHandler(self._write_log, asyncio.get_running_loop())
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

        # 对话区只看事件：用户的话以识别结果为准，语音模式下也一样能显示
        conversation = self.query_one(ConversationView)
        bus = self._backend.bus
        self._unsubscribe = [
            bus.subscribe(Transcribed, lambda e: conversation.add_user(e.text)),
            bus.subscribe(Spoken, lambda e: conversation.add_spoken(e.text, e.utterance)),
            bus.subscribe(ChoicesShown, lambda e: conversation.add_choices(e.lines)),
        ]
        self.run_worker(self._run_backend(), name="pinecone")
        self.run_worker(self._enable_input(), name="ready")

    def on_unmount(self) -> None:
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)

    async def action_quit(self) -> None:
        self._backend.stop()

    async def action_interrupt(self) -> None:
        """像说“停止”一样打断正在思考或播报的这一轮，后台任务不受影响。"""
        if self._backend.ready.is_set():
            await self._backend.dialog.interrupt("user")

    def action_clear(self) -> None:
        self.query_one(ConversationView).clear()
        self.query_one(LogView).clear_records()

    def action_toggle_log(self) -> None:
        self.query_one(LogView).toggle_class("hidden")

    def action_warnings_only(self) -> None:
        self.query_one(LogView).toggle_warnings_only()
        self._update_log_title()

    def _update_log_title(self) -> None:
        log = self.query_one(LogView)
        log.border_title = "日志 · 只看警告和错误" if log.min_level else "日志"

    async def _run_backend(self) -> None:
        try:
            await self._backend.run(stdin=False)
        finally:
            self.exit()

    async def _enable_input(self) -> None:
        await self._backend.ready.wait()
        prompt = self.query_one(PromptInput)
        prompt.placeholder = "输入一句话后回车（exit 退出）"
        prompt.disabled = False
        prompt.focus()
        backend = self._backend
        self.query_one(StatusBar).watch(backend.state, backend.dialog, backend.tasks)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        self.query_one(PromptInput).remember(text)
        if text in ("exit", "quit"):
            self._backend.stop()
            return
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
            log = self.query_one(LogView)
        except NoMatches:  # 界面已经关闭，投递过来的最后几条日志在文件里
            return
        log.add(level, text)
