"""输入框：上下键翻历史，历史存到数据目录，下次启动还在。"""

from __future__ import annotations

import logging
from pathlib import Path

from textual.binding import Binding
from textual.widgets import Input

logger = logging.getLogger(__name__)

_MAX_HISTORY = 500


class PromptInput(Input):
    """带历史记录的输入框。

    ↑ 往前翻、↓ 往后翻，翻到最新一条之后回到正在输入的内容；连续重复的一句只记一次。
    """

    BINDINGS = [
        Binding("up", "history(-1)", "上一句", show=False),
        Binding("down", "history(1)", "下一句", show=False),
    ]

    def __init__(self, history_file: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._file = history_file
        self._history = self._load()
        self._index = len(self._history)
        self._draft = ""

    def remember(self, text: str) -> None:
        """记下一句已发送的话，并回到“正在输入”的位置。"""
        if not self._history or self._history[-1] != text:
            self._history.append(text)
            del self._history[:-_MAX_HISTORY]
            self._save()
        self._index = len(self._history)
        self._draft = ""

    def action_history(self, step: int) -> None:
        if not self._history:
            return
        if self._index == len(self._history):
            self._draft = self.value
        self._index = min(max(self._index + step, 0), len(self._history))
        self.value = (
            self._draft if self._index == len(self._history) else self._history[self._index]
        )
        self.cursor_position = len(self.value)

    def _load(self) -> list[str]:
        try:
            lines = self._file.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except OSError:
            logger.warning("读不了输入历史：%s", self._file, exc_info=True)
            return []
        return [line for line in lines if line][-_MAX_HISTORY:]

    def _save(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text("\n".join(self._history) + "\n", encoding="utf-8")
        except OSError:
            logger.warning("保存输入历史失败：%s", self._file, exc_info=True)
