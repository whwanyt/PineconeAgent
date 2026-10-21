"""Agent 测试用的假模型：按剧本依次返回消息，支持流式、工具调用和工具绑定。"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk


class ScriptedChatModel(GenericFakeChatModel):
    """按顺序返回预设消息的假模型，并记录每次收到的消息。"""

    received: list[list[BaseMessage]] = []
    bound_tools: list[str] = []

    @classmethod
    def from_script(cls, *replies: AIMessage | str | Exception) -> ScriptedChatModel:
        def gen() -> Iterator[AIMessage]:
            for reply in replies:
                if isinstance(reply, Exception):
                    raise reply
                yield AIMessage(content=reply) if isinstance(reply, str) else reply

        return cls(messages=gen(), received=[], bound_tools=[])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def _stream(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        message = self._generate(messages, *args, **kwargs).generations[0].message
        assert isinstance(message, AIMessage)
        content = message.content if isinstance(message.content, str) else ""
        # 文本按空白切成多个片段，模拟逐 token 输出
        for token in [t for t in re.split(r"(\s)", content) if t]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=token, id=message.id))
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    id=message.id,
                    tool_call_chunks=[
                        {
                            "name": c["name"],
                            "args": json.dumps(c["args"]),
                            "id": c["id"],
                            "index": i,
                        }
                        for i, c in enumerate(message.tool_calls)
                    ],
                )
            )
