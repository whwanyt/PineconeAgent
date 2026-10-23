"""把 C/C++ 库直接写到终端的输出转进日志。

sherpa-onnx、PortAudio 这类库不走 Python 的 ``sys.stderr``，直接往文件描述符 1、2 写，
全屏界面下这些字会印在界面上。这里先复制一份终端留给界面用，再把 1、2 号接到管道上，
读出来按行写日志，于是它们出现在日志区里，也进日志文件。
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Iterator

logger = logging.getLogger("native")


@contextlib.contextmanager
def native_output_to_log() -> Iterator[None]:
    """在这段时间里，把直接写 fd 1、2 的输出转成 ``native`` 日志。

    必须在创建界面之前进入：Textual 在创建时记下 ``sys.__stderr__``，之后都往那里画界面。
    """
    saved_fds = (os.dup(1), os.dup(2))
    saved_streams = (sys.__stdout__, sys.__stderr__)
    # 界面画在这份终端副本上，不受下面接管 1、2 号的影响
    terminal = os.fdopen(os.dup(saved_fds[1]), "w", encoding="utf-8", buffering=1)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    sys.__stdout__ = sys.__stderr__ = terminal
    reader = threading.Thread(target=_forward, args=(read_fd,), name="native-output", daemon=True)
    reader.start()
    try:
        yield
    finally:
        sys.__stdout__, sys.__stderr__ = saved_streams
        # 还原 1、2 号后管道没有写端了，读线程读到结尾自己退出
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        for fd in saved_fds:
            os.close(fd)
        reader.join(timeout=1)
        terminal.close()


def _forward(read_fd: int) -> None:
    with os.fdopen(read_fd, "rb") as pipe:
        for raw in pipe:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                logger.info("%s", line)
