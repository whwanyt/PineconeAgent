"""对话区：用户说的话、音箱播报的话、候选列表，按时间顺序排下来。"""

from __future__ import annotations

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

_MAX_MESSAGES = 300
"""最多保留多少条，太早的移除，长时间调试也不会越来越卡。"""


class Message(Static):
    """一条消息：左边是说话人，右边是内容。"""

    DEFAULT_CSS = """
    Message {
        margin: 0 0 1 0;
    }
    Message.choices {
        margin: 0 0 1 4;
        color: $text-muted;
    }
    """

    def __init__(self, speaker: str, style: str, text: str, *, classes: str = "") -> None:
        super().__init__(classes=classes)
        self.speaker = speaker
        self.label_style = style
        self.body = text
        self._refresh_content()

    def append(self, text: str) -> None:
        self.body += text
        self._refresh_content()

    def _refresh_content(self) -> None:
        line = Text()
        if self.speaker:
            line.append(f"{self.speaker}  ", style=self.label_style)
        line.append(self.body)
        self.update(line)


class ConversationView(VerticalScroll):
    """对话区。

    同一次播报的各句（``Spoken.utterance`` 相同）拼成一条，流式回复会一句句长出来；
    滚到最底部时新消息自动跟随，往上翻看历史时不打扰。
    """

    def __init__(self, assistant: str, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._assistant = assistant
        self._utterance: int | None = None
        self._speaking: Message | None = None

    def on_mount(self) -> None:
        self.anchor()

    def add_user(self, text: str) -> None:
        self._add(Message("你", "bold cyan", text))

    def add_spoken(self, text: str, utterance: int) -> None:
        if self._speaking is not None and utterance == self._utterance:
            self._speaking.append(text)
            return
        self._utterance = utterance
        self._speaking = self._add(Message(self._assistant, "bold green", text))

    def add_choices(self, lines: str) -> None:
        self._add(Message("", "", lines, classes="choices"))

    @property
    def messages(self) -> list[Message]:
        return list(self.query(Message))

    def _add(self, message: Message) -> Message:
        if message.speaker != self._assistant:
            # 别人插话之后，同一编号也不再往上一条播报里拼
            self._speaking = None
        self.mount(message)
        extra = len(self.children) - _MAX_MESSAGES
        for old in list(self.children)[: max(extra, 0)]:
            old.remove()
        return message
