"""命令行入口。

用法::

    python -m pinecone.main                 # 语音模式
    python -m pinecone.main --text          # 文本模式（终端输入代替说话）
    python -m pinecone.main -c my.yaml      # 指定配置文件
    pinecone                                # 安装后的等价命令
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from pinecone.app import App
from pinecone.core.config import load_config
from pinecone.core.logging import setup_logging

logger = logging.getLogger("pinecone")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pinecone", description="树莓派智能音箱")
    parser.add_argument("-c", "--config", help="配置文件路径（默认 config/config.yaml）")
    parser.add_argument("--text", action="store_true", help="文本模式：终端输入代替语音")
    parser.add_argument("--debug", action="store_true", help="输出 DEBUG 日志")
    return parser.parse_args()


async def _amain(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.debug:
        config.log.level = "DEBUG"
    setup_logging(config)

    app = App(config, text_mode=args.text)
    # systemd 停止服务时发送 SIGTERM，Ctrl+C 发送 SIGINT，两者都走优雅退出流程
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, app.stop)
    await app.run()


def main() -> None:
    """程序入口。"""
    asyncio.run(_amain(parse_args()))


if __name__ == "__main__":
    main()
