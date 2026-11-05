"""命令行：只有正常和 debug 两种启动，全屏界面取决于终端和 textual。"""

from __future__ import annotations

import sys

import pytest

from pinecone.main import _want_tui, parse_args


def test_parse_args_keeps_debug_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """只接受配置路径和 debug，不再有单独的文本模式开关。"""
    monkeypatch.setattr(sys, "argv", ["pinecone", "-c", "my.yaml", "--debug"])
    args = parse_args()
    assert args.config == "my.yaml" and args.debug is True
    assert not hasattr(args, "text") and not hasattr(args, "plain")


def test_unknown_text_flag_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """旧的 --text 不再是合法参数。"""
    monkeypatch.setattr(sys, "argv", ["pinecone", "--text"])
    with pytest.raises(SystemExit):
        parse_args()


def test_tui_requires_tty_and_textual(monkeypatch: pytest.MonkeyPatch) -> None:
    """两端都是终端且装了 textual 才用全屏界面；缺终端时不警告。"""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("pinecone.main.importlib.util.find_spec", lambda name: object())
    assert _want_tui() == (True, None)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert _want_tui() == (False, None)


def test_missing_textual_explains_why(monkeypatch: pytest.MonkeyPatch) -> None:
    """有终端但没装 textual 时说明原因，调用方用来打警告。"""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("pinecone.main.importlib.util.find_spec", lambda name: None)
    use, why = _want_tui()
    assert use is False and why is not None and "textual" in why
