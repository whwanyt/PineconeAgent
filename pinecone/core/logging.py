"""日志初始化。

统一格式：``时间 级别 [模块] 消息``。同时输出到控制台和滚动日志文件，
在树莓派上用 systemd 运行时，控制台输出会进入 journald（``journalctl -u pinecone``）。
文本界面占用整个终端，此时关掉控制台输出，由界面自己挂 handler 显示日志。
"""

from __future__ import annotations

import logging
import logging.handlers

from pinecone.core.config import AppConfig

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# 这些第三方库在 INFO 级别下输出过多，统一调到 WARNING
_NOISY_LOGGERS = ("httpx", "httpcore", "apscheduler", "openai", "asyncio", "aiosqlite")


def setup_logging(config: AppConfig, *, console: bool = True) -> None:
    """根据配置初始化根日志器。

    可以重复调用（例如测试中），每次调用都会先清空已有 handler，避免日志重复输出。

    Args:
        config: 应用配置，使用其中的 ``log`` 段。
        console: 是否输出到控制台。全屏文本界面运行时关闭，避免日志写乱界面。
    """
    log_cfg = config.log
    root = logging.getLogger()
    root.setLevel(log_cfg.level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)

    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_cfg.file:
        log_path = config.resolve_path(log_cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # RotatingFileHandler：防止日志无限增长占满树莓派的 SD 卡
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=log_cfg.max_bytes,
            backupCount=log_cfg.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
