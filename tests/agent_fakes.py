"""Agent 测试用的假模型：按剧本依次返回消息，支持流式和工具绑定。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage


class ScriptedChatModel(GenericFakeChatModel):
    """按顺序返回预设消息的假模型，并记录每次收到的消息。"""

    received: list[list[BaseMessage]] = []

    @classmethod
    def from_script(cls, *replies: AIMessage | str | Exception) -> ScriptedChatModel:
        def gen() -> Iterator[AIMessage]:
            for reply in replies:
                if isinstance(reply, Exception):
                    raise reply
                yield AIMessage(content=reply) if isinstance(reply, str) else reply

        return cls(messages=gen(), received=[])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)
