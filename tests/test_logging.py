"""日志初始化测试。"""

import logging
import logging.handlers
from pathlib import Path

from pinecone.core.config import AppConfig
from pinecone.core.logging import setup_logging


def test_console_can_be_turned_off(tmp_path: Path) -> None:
    """全屏界面运行时不挂控制台 handler，日志仍写文件；重复调用不会叠加 handler。"""
    config = AppConfig(data_dir=str(tmp_path))
    config.log.file = str(tmp_path / "pinecone.log")
    root = logging.getLogger()
    saved = list(root.handlers)
    try:
        setup_logging(config)
        setup_logging(config, console=False)
        kinds = [type(h) for h in root.handlers]
        assert kinds == [logging.handlers.RotatingFileHandler]
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved:
            root.addHandler(handler)
