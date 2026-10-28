"""命令行入口。

用法::

    python -m pinecone.main                 # 语音模式
    python -m pinecone.main --text          # 文本模式（全屏界面，终端输入代替说话）
    python -m pinecone.main --text --plain  # 文本模式，逐行读标准输入（可以用管道喂指令）
    python -m pinecone.main -c my.yaml      # 指定配置文件
    pinecone                                # 安装后的等价命令
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import signal
import sys

from pinecone.app import App
from pinecone.core.config import load_config
from pinecone.core.logging import setup_logging

logger = logging.getLogger("pinecone")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pinecone", description="树莓派智能音箱")
    parser.add_argument("-c", "--config", help="配置文件路径（默认 config/config.yaml）")
    parser.add_argument("--text", action="store_true", help="文本模式：终端输入代替语音")
    parser.add_argument("--plain", action="store_true", help="文本模式不用全屏界面，逐行读标准输入")
    parser.add_argument("--debug", action="store_true", help="输出 DEBUG 日志")
    return parser.parse_args()


def _want_tui(args: argparse.Namespace) -> tuple[bool, str | None]:
    """文本模式是否用全屏界面。返回（是否使用，不能用时的原因）。"""
    if not args.text or args.plain:
        return False, None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False, None
    if importlib.util.find_spec("textual") is None:
        return False, "没有安装 textual，改用逐行输入（安装：uv sync --extra tui）"
    return True, None


async def _amain(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.debug:
        config.log.level = "DEBUG"
    use_tui, why_not = _want_tui(args)
    # 全屏界面占满终端，日志改由界面显示，控制台不再输出
    setup_logging(config, console=not use_tui)
    if why_not:
        logger.warning(why_not)

    app = App(config, text_mode=args.text)
    # systemd 停止服务时发送 SIGTERM，Ctrl+C 发送 SIGINT，两者都走优雅退出流程。
    # 全屏界面下终端是原始模式，Ctrl+C 是按键，由界面处理
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, app.stop)
    if use_tui:
        from pinecone.tui import PineconeTui
        from pinecone.tui.native import native_output_to_log

        with native_output_to_log():
            await PineconeTui(app.session).run_async()
    else:
        await app.run()


def main() -> None:
    """程序入口。"""
    asyncio.run(_amain(parse_args()))


if __name__ == "__main__":
    main()
