"""语音 AI Agent（LangChain ``create_agent`` + LangGraph）。

在整体架构中的位置::

    DialogController ─▶ IntentRouter ──规则未命中──▶ VoiceAgent.handle(text)
                                                     │
                                                     ▼  Reply(stream=逐句文本)
                              Speaker.say_stream ◀── _stream()：graph.astream(stream_mode="messages")

流式按句播报：
    ``astream(stream_mode="messages")`` 逐 token 产出模型输出，这里只挑出 **模型节点** 的文本
    片段交给 Speaker，Speaker 内部按标点切句、边合成边播放，首句延迟约等于“首个整句生成时间”。
    工具调用产生的消息不会被念出来。

打断与取消：
    整个流运行在「对话」托管任务中。被唤醒词打断或说“停止”时，任务被取消，
    ``CancelledError`` 会沿着异步生成器传进 LangGraph，模型请求随之中断。
    如果恰好打断在工具调用之后、工具结果之前，会话里会残留“没有结果的工具调用”，
    下一轮开始前由 :meth:`VoiceAgent._repair_history` 补上“已取消”的工具结果，
    否则 OpenAI 兼容接口会直接报错。

会话（短期记忆）：
    语音场景没有“新建对话”按钮，这里按空闲时间切分会话：距上次对话超过
    ``session_idle_min`` 分钟就换一个新的 thread_id。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, dynamic_prompt
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore

from pinecone.core.config import AppConfig
from pinecone.dialog import Reply

from .prompts import build_system_prompt

logger = logging.getLogger(__name__)

_ERROR_REPLY = "抱歉，我现在连不上大模型，请稍后再试。"


@dataclass
class AgentContext:
    """每次调用时传给 Agent 的运行时上下文（不会写入会话历史）。"""

    memories: list[str] = field(default_factory=list)
    """本次对话前检索到的长期记忆。"""


class Session:
    """按空闲时间切分会话，提供当前的 thread_id。"""

    def __init__(self, idle_min: float) -> None:
        self._idle_s = idle_min * 60
        self._thread_id = ""
        self._last_active = 0.0

    def thread_id(self) -> str:
        """当前会话 id；空闲超时则开启新会话。"""
        now = time.time()
        if not self._thread_id or now - self._last_active > self._idle_s:
            self._thread_id = f"voice-{int(now)}"
            logger.info("开启新会话：%s", self._thread_id)
        self._last_active = now
        return self._thread_id

    def reset(self) -> None:
        """强制下次开启新会话（例如用户说“换个话题”）。"""
        self._thread_id = ""


def _text_of(chunk: AIMessageChunk) -> str:
    """取出消息片段中的文本（部分模型的 content 是分段列表）。"""
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


class VoiceAgent:
    """实现 :class:`~pinecone.dialog.DialogHandler` 协议的 AI 对话处理器。"""

    def __init__(
        self,
        config: AppConfig,
        model: BaseChatModel,
        *,
        tools: Sequence[BaseTool] = (),
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        middleware: Sequence[AgentMiddleware] = (),
    ) -> None:
        self._config = config
        self.session = Session(config.agent.session_idle_min)
        self._store = store
        name, persona = config.name, config.agent.persona

        @dynamic_prompt
        def system_prompt(request: ModelRequest) -> str:
            # 每次调用模型时重新生成：当前时间会变，长期记忆来自运行时上下文
            context = request.runtime.context
            memories = context.memories if isinstance(context, AgentContext) else []
            return build_system_prompt(name, persona, memories)

        self._graph = create_agent(
            model,
            tools=list(tools),
            middleware=[system_prompt, *middleware],
            context_schema=AgentContext,
            checkpointer=checkpointer or InMemorySaver(),
            store=store,
        )

    async def handle(self, text: str) -> Reply:
        """处理一句用户文本，返回流式回复（真正的模型调用在播报时才开始）。"""
        return Reply(stream=self._stream(text), follow_up=True)

    async def _stream(self, text: str) -> AsyncIterator[str]:
        config: dict[str, Any] = {"configurable": {"thread_id": self.session.thread_id()}}
        produced = False
        try:
            await self._repair_history(config)
            context = AgentContext(memories=await self._load_memories(text))
            async for chunk, metadata in self._graph.astream(
                {"messages": [{"role": "user", "content": text}]},
                config=config,
                context=context,
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") != "model" or not isinstance(
                    chunk, AIMessageChunk
                ):
                    continue
                piece = _text_of(chunk)
                if piece:
                    produced = True
                    yield piece
        except Exception:
            logger.exception("AI 对话失败")
            # 已经说了一半时不再追加道歉，避免听起来很奇怪
            if not produced:
                yield _ERROR_REPLY

    async def _load_memories(self, text: str) -> list[str]:
        """检索与本轮相关的长期记忆（未启用长期记忆时为空）。"""
        return []

    async def _repair_history(self, config: dict[str, Any]) -> None:
        """给被打断的工具调用补上结果，保证会话历史合法。"""
        state = await self._graph.aget_state(config)
        messages = state.values.get("messages", []) if state and state.values else []
        if not messages:
            return
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            missing = [c for c in last.tool_calls if c["id"] not in answered]
            if missing:
                logger.info("补全被打断的工具调用：%s", [c["name"] for c in missing])
                await self._graph.aupdate_state(
                    config,
                    {
                        "messages": [
                            ToolMessage(content="已被用户打断取消", tool_call_id=c["id"])
                            for c in missing
                        ]
                    },
                    # 没有工具时图里不存在 tools 节点，以 model 节点的名义写入即可
                    as_node="model",
                )
