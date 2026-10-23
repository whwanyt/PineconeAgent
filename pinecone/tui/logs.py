"""日志区：把日志从各线程转给界面，按级别着色显示。"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable

from rich.text import Text
from textual.widgets import RichLog

LogSink = Callable[[int, str], None]
"""接收一条日志：级别号 + 格式化好的文本。只在事件循环线程中调用。"""


class TuiLogHandler(logging.Handler):
    """把日志记录投递回事件循环，交给界面显示。

    日志可能来自任何线程（音频回调、``to_thread`` 里的模型推理），界面控件只能在
    事件循环线程里改，所以一律经 ``call_soon_threadsafe`` 转一手。
    """

    def __init__(self, sink: LogSink, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._sink = sink
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:
            self.handleError(record)
            return
        # 界面退出、事件循环关闭后还有零星日志：日志文件里有，这里丢掉即可
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._sink, record.levelno, text)


_LEVEL_STYLES = {
    logging.DEBUG: "dim",
    logging.INFO: "",
    logging.WARNING: "yellow",
    logging.ERROR: "bold red",
    logging.CRITICAL: "bold white on red",
}

_MAX_RECORDS = 2000


class LogView(RichLog):
    """日志区：按级别着色，可以只看警告和错误。

    收到的日志都留一份（最多 2000 条），切换过滤时按新条件重画，之前的日志不会丢。
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, wrap=True, markup=False, max_lines=_MAX_RECORDS)
        self._records: deque[tuple[int, str]] = deque(maxlen=_MAX_RECORDS)
        self.min_level = logging.NOTSET

    def add(self, level: int, text: str) -> None:
        self._records.append((level, text))
        if level >= self.min_level:
            self._write(level, text)

    def toggle_warnings_only(self) -> None:
        self.min_level = logging.NOTSET if self.min_level else logging.WARNING
        self.clear()
        for level, text in self._records:
            if level >= self.min_level:
                self._write(level, text)

    def _write(self, level: int, text: str) -> None:
        style = _LEVEL_STYLES.get(level, _LEVEL_STYLES[logging.CRITICAL])
        self.write(Text(text, style=style))
