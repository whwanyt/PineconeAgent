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

模型提问：
    只有模型调用 ``ask_user`` 时才会 ``interrupt``。流结束后如果检查点上还有未回答的提问，
    把问句接在模型已经说出的话后面。下一句能解析成选项或取消时，用
    ``Command(resume=用户原话)`` 沿用同一个 thread_id 继续。
    已经暂停的提问不能补假的工具结果，否则无法恢复。

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
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from pinecone.agent.question import (
    AnswerKind,
    PendingQuestion,
    omit_repeated_lead,
    parse_answer,
    pending_from_value,
)
from pinecone.agent.tools.ask_user import ask_user_tool
from pinecone.core.config import AppConfig
from pinecone.dialog import Reply

from .memory.long_term import LongTermMemory
from .prompts import build_system_prompt

logger = logging.getLogger(__name__)

_ERROR_REPLY = "抱歉，我现在连不上大模型，请稍后再试。"


@dataclass
class AgentContext:
    """每次调用时传给 Agent 的运行时上下文（不会写入会话历史）。"""

    memories: list[str] = field(default_factory=list)
    """本次对话前检索到的长期记忆。"""
    hint: str = ""
    """技能转交时的情况说明（例如本地曲库没找到）。只注入本轮系统提示。"""


class Session:
    """按空闲时间切分会话，提供当前的 thread_id。"""

    def __init__(self, idle_min: float) -> None:
        self._idle_s = idle_min * 60
        self._thread_id = ""
        self._last_active = 0.0

    def current(self) -> str:
        """当前 thread_id。还没有会话时为空字符串，不会因此新建。"""
        return self._thread_id

    def thread_id(self) -> str:
        """当前会话 id；空闲超时则开启新会话。"""
        now = time.time()
        if not self._thread_id or now - self._last_active > self._idle_s:
            self._thread_id = f"voice-{int(now)}"
            logger.info("开启新会话：%s", self._thread_id)
        self._last_active = now
        return self._thread_id

    def hold(self) -> str:
        """沿用当前会话并刷新空闲计时。有未完成提问时调用，避免打字或迟疑被切成新会话。"""
        now = time.time()
        self._last_active = now
        if not self._thread_id:
            self._thread_id = f"voice-{int(now)}"
            logger.info("开启新会话：%s", self._thread_id)
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
        memory: LongTermMemory | None = None,
        middleware: Sequence[AgentMiddleware] = (),
    ) -> None:
        self._config = config
        self.session = Session(config.agent.session_idle_min)
        self._memory = memory
        name, persona = config.name, config.agent.persona
        all_tools = [
            ask_user_tool(config.agent.choice_limit),
            *tools,
            *(memory.tools() if memory else []),
        ]

        @dynamic_prompt
        def system_prompt(request: ModelRequest) -> str:
            # 每次调用模型时重新生成：当前时间、长期记忆、本轮 hint 都可能变
            context = request.runtime.context
            memories = context.memories if isinstance(context, AgentContext) else []
            hint = context.hint if isinstance(context, AgentContext) else ""
            return build_system_prompt(name, persona, memories, hint=hint)

        self._graph = create_agent(
            model,
            tools=all_tools,
            middleware=[system_prompt, *middleware],
            context_schema=AgentContext,
            checkpointer=checkpointer or InMemorySaver(),
        )
        logger.info("AI 工具：%s", "、".join(t.name for t in all_tools) or "无")

    async def handle(self, text: str, *, hint: str | None = None) -> Reply:
        """处理一句用户文本，返回流式回复（真正的模型调用在播报时才开始）。

        有未完成的 ``ask_user`` 时，这句是回答而不是新的用户消息：能解析就恢复图，
        听不清则再把问句说一遍。改口由路由器在调用本方法前丢掉这次提问。

        Args:
            text: 用户原话。没有提问时写入会话历史；正在提问时作为 ``interrupt`` 的返回值。
            hint: 技能转交时的情况说明，只进本轮系统提示，不写进用户消息。
        """
        reply = Reply(follow_up=True)
        pending = await self.pending_question()
        if pending is not None:
            answer = parse_answer(text, pending.choices, limit=pending.limit)
            if answer.kind is AnswerKind.UNCLEAR:
                self._mark_question(reply, pending)
                reply.text = pending.prompt
                return reply
            reply.stream = self._drive(Command(resume=text), reply, memory_text=text, repair=False)
            return reply
        reply.stream = self._drive(
            {"messages": [{"role": "user", "content": text}]},
            reply,
            memory_text=text,
            hint=hint,
            repair=True,
        )
        return reply

    async def pending_question(self) -> PendingQuestion | None:
        """当前会话里还没回答的提问。没有会话或已经回答过时返回 None。"""
        thread = self.session.current()
        if not thread:
            return None
        state = await self._graph.aget_state({"configurable": {"thread_id": thread}})
        if state is None:
            return None
        for item in state.interrupts:
            pending = pending_from_value(getattr(item, "value", None))
            if pending is not None:
                return pending
        return None

    def drop_pending(self) -> None:
        """放弃未完成的提问并换一个新会话。旧检查点留在库里，不再恢复。"""
        self.session.reset()

    def _mark_question(self, reply: Reply, pending: PendingQuestion) -> None:
        """让对话层用更长的追问窗口，并在文本模式打印编号。"""
        reply.follow_up = True
        reply.listen_s = self._config.agent.choice_timeout_s
        reply.choice_lines = pending.lines

    async def _run_config(self) -> dict[str, Any]:
        """本轮使用的 thread 配置。有提问时不因空闲超时换会话。"""
        if await self.pending_question() is not None:
            thread = self.session.hold()
        else:
            thread = self.session.thread_id()
        return {"configurable": {"thread_id": thread}}

    async def _drive(
        self,
        graph_input: dict[str, Any] | Command,
        reply: Reply,
        *,
        memory_text: str,
        repair: bool,
        hint: str | None = None,
    ) -> AsyncIterator[str]:
        """跑一轮图，产出要念的文本。模型提问暂停时，问句接在已生成文本的后面。"""
        config = await self._run_config()
        produced = False
        spoken: list[str] = []
        try:
            if repair:
                await self._repair_history(config)
            context = AgentContext(memories=await self._load_memories(memory_text), hint=hint or "")
            async for chunk, metadata in self._graph.astream(
                graph_input,
                config=config,
                context=context,
                stream_mode="messages",
                durability="sync",
            ):
                if metadata.get("langgraph_node") != "model" or not isinstance(
                    chunk, AIMessageChunk
                ):
                    continue
                piece = _text_of(chunk)
                if piece:
                    produced = True
                    spoken.append(piece)
                    yield piece
            pending = await self.pending_question()
            if pending is not None:
                self._mark_question(reply, pending)
                extra = omit_repeated_lead("".join(spoken), pending.prompt)
                if extra:
                    produced = True
                    yield extra
        except GraphInterrupt:
            pending = await self.pending_question()
            if pending is not None:
                self._mark_question(reply, pending)
                extra = omit_repeated_lead("".join(spoken), pending.prompt)
                if extra:
                    yield extra
                return
            logger.exception("AI 对话失败")
            if not produced:
                yield _ERROR_REPLY
        except Exception:
            logger.exception("AI 对话失败")
            if not produced:
                yield _ERROR_REPLY

    async def _load_memories(self, text: str) -> list[str]:
        """检索与本轮相关的长期记忆（未启用长期记忆时为空）。"""
        return await self._memory.for_prompt(text) if self._memory else []

    async def _repair_history(self, config: dict[str, Any]) -> None:
        """给被打断的工具调用补上结果，保证会话历史合法。

        已经干净暂停、正在等用户回答的提问不能补假结果，否则恢复时对不上工具调用。
        """
        state = await self._graph.aget_state(config)
        if state is not None and any(
            pending_from_value(getattr(item, "value", None)) is not None
            for item in state.interrupts
        ):
            return
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
