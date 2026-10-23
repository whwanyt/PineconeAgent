"""把日志转给界面显示的 handler。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

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
