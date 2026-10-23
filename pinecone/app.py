"""应用组装：创建各模块实例，按正确顺序接线、启动和关闭。

模块依赖关系（自下而上）::

    EventBus ─┬─ StateMachine
              ├─ TaskManager
              ├─ AudioInput ──(帧)──▶ SpeechPipeline（唤醒词 + VAD）
              ├─ AudioOutput ◀── Speaker ◀── TTS
              ├─ EchoCanceller（连接 AudioInput 与 AudioOutput）
              └─ DialogController（订阅事件，驱动整个对话流程）
                    └─ IntentRouter ─▶ SkillManager（规则匹配技能）
                                   └─▶ VoiceAgent（规则未命中时的 AI 兜底）

两种运行模式：
    - 语音模式（默认）：麦克风 + 唤醒词 + 识别；
    - 文本模式（``--text``）：从终端输入文字代替说话，方便在电脑上调试技能和 AI，
      仍然会通过扬声器播报回复。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from pinecone.agent import VoiceAgent, create_chat_model
from pinecone.agent.memory.long_term import LongTermMemory
from pinecone.agent.memory.short_term import create_summarization, prune_history
from pinecone.agent.tools.mcp_tools import load_mcp_tools
from pinecone.agent.tools.skill_tools import skill_tools
from pinecone.audio.aec import create_echo_canceller
from pinecone.audio.input import AudioInput
from pinecone.audio.output import AudioOutput
from pinecone.core.bus import EventBus
from pinecone.core.config import AppConfig
from pinecone.core.events import ChoicesShown
from pinecone.core.state import StateMachine
from pinecone.core.task_manager import TaskManager
from pinecone.dialog import DialogController, EchoHandler
from pinecone.nlu.router import IntentRouter
from pinecone.skills import SkillContext, SkillManager
from pinecone.speech.tts import Speaker, create_tts

logger = logging.getLogger(__name__)


class App:
    """PineconeAgent 应用。"""

    def __init__(self, config: AppConfig, *, text_mode: bool = False) -> None:
        self.config = config
        self.text_mode = text_mode
        self.bus = EventBus()
        self.state = StateMachine(self.bus)
        self.tasks = TaskManager(self.bus)
        self.output = AudioOutput(config.audio)
        self.speaker = Speaker(create_tts(config), self.output, self.bus)
        self.mic: AudioInput | None = None
        self.dialog: DialogController
        self._stop = asyncio.Event()
        # 需要在退出时关闭的异步资源（数据库连接、MCP 会话等）
        self._resources = contextlib.AsyncExitStack()

    def _build_voice_frontend(self) -> DialogController:
        """创建麦克风、回声消除、唤醒词、VAD、识别，并组装对话控制器。"""
        # 延迟导入：文本模式下不需要加载语音模型
        from pinecone.speech.asr import create_recognizer
        from pinecone.speech.pipeline import SpeechPipeline
        from pinecone.speech.vad.silero import SileroVad
        from pinecone.speech.wake.sherpa_kws import SherpaKeywordSpotter

        cfg = self.config
        self.mic = AudioInput(cfg.audio)
        aec = create_echo_canceller(cfg.audio)
        aec.attach(self.mic, self.output)
        logger.info("回声消除模式：%s（%s）", aec.mode, "生效" if aec.effective else "未生效")

        wake = SherpaKeywordSpotter(
            cfg.wake, cfg.resolve_path(cfg.wake.model_dir), cfg.audio.sample_rate
        )
        vad = SileroVad(cfg.vad, cfg.resolve_path(cfg.vad.model), cfg.audio.sample_rate)
        pipeline = SpeechPipeline(self.bus, wake, vad)
        self.mic.add_listener(pipeline.on_frame)

        return DialogController(
            config=cfg,
            bus=self.bus,
            state=self.state,
            tasks=self.tasks,
            speaker=self.speaker,
            output=self.output,
            handler=EchoHandler(),
            pipeline=pipeline,
            asr=create_recognizer(cfg),
            aec_effective=aec.effective,
        )

    def _build_text_frontend(self) -> DialogController:
        return DialogController(
            config=self.config,
            bus=self.bus,
            state=self.state,
            tasks=self.tasks,
            speaker=self.speaker,
            output=self.output,
            handler=EchoHandler(),
        )

    def _make_skill_context(self, skill_name: str) -> SkillContext:
        """为每个技能创建专属上下文。"""
        ctx = SkillContext(
            skill_name=skill_name,
            config=self.config,
            bus=self.bus,
            tasks=self.tasks,
            dialog=self.dialog,
            output=self.output,
        )
        ctx.extras["skills"] = self.skills
        return ctx

    async def _build_agent(self) -> VoiceAgent | None:
        """创建 AI Agent；未启用或没有配置密钥时返回 None（规则未命中时回复“我还不会”）。"""
        cfg = self.config.agent
        if not cfg.enabled:
            logger.info("AI 对话已关闭（agent.enabled=false）")
            return None
        if not cfg.api_key:
            logger.warning("没有配置 agent.api_key / OPENAI_API_KEY，AI 对话不可用")
            return None
        model = create_chat_model(cfg)

        # 短期记忆：SQLite 检查点，连接在整个运行期间保持打开，退出时由 _resources 关闭
        history_db = self.config.data_path / "agent_history.db"
        prune_history(history_db, cfg.history_days)
        checkpointer = await self._resources.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(history_db))
        )
        summarization = create_summarization(
            model,
            summarize_after_tokens=cfg.summarize_after_tokens,
            keep_messages=cfg.keep_messages,
        )

        # 长期记忆：用户偏好与事实，跨会话保留
        memory = None
        if cfg.long_term_memory:
            store = await self._resources.enter_async_context(
                AsyncSqliteStore.from_conn_string(str(self.config.data_path / "agent_memory.db"))
            )
            await store.setup()
            memory = LongTermMemory(store)

        agent = VoiceAgent(
            self.config,
            model,
            tools=[
                *skill_tools(self.skills),
                *await load_mcp_tools(cfg.mcp_servers, call_timeout_s=cfg.mcp_tool_timeout_s),
            ],
            checkpointer=checkpointer,
            memory=memory,
            middleware=[summarization],
        )
        logger.info("AI 对话已启用：%s @ %s", cfg.model, cfg.base_url)
        return agent

    async def setup(self) -> None:
        """创建并启动各模块（不阻塞）。"""
        self.bus.bind_loop(asyncio.get_running_loop())
        self.dialog = (
            self._build_text_frontend() if self.text_mode else self._build_voice_frontend()
        )

        # 技能：加载完成后用路由器替换调试用的 EchoHandler
        self.skills = SkillManager(self.config, self.tasks, self._make_skill_context)
        await self.skills.load()
        self.router = IntentRouter(self.skills)
        self.router.set_fallback(await self._build_agent())
        self.dialog.set_handler(self.router)

        self.dialog.start()
        self.output.start()
        if self.mic is not None:
            self.mic.start()
        wake_words = "、".join(self.config.wake.keywords)
        logger.info(
            "%s 已启动%s",
            self.config.name,
            "（文本模式）" if self.text_mode else f"，请说「{wake_words}」唤醒",
        )

    async def run(self) -> None:
        """启动并运行，直到 :meth:`stop` 被调用。"""
        await self.setup()
        text_loop = asyncio.create_task(self._text_loop()) if self.text_mode else None
        try:
            await self._stop.wait()
        finally:
            if text_loop is not None:
                text_loop.cancel()
            await self.shutdown()

    def stop(self) -> None:
        """请求退出（可在信号处理函数中调用）。"""
        self._stop.set()

    async def shutdown(self) -> None:
        """按与启动相反的顺序关闭：先停输入，再取消任务，最后关输出。"""
        logger.info("正在退出……")
        if self.mic is not None:
            self.mic.stop()
        await self.tasks.shutdown()
        await self.skills.unload()
        await self._resources.aclose()
        self.output.stop()

    async def _text_loop(self) -> None:
        """文本模式：从标准输入逐行读取，当作识别结果处理。输入 exit 退出。"""
        print("文本模式：输入一句话后回车（exit 退出）")
        self.bus.subscribe(ChoicesShown, lambda e: print(e.lines, flush=True))
        while True:
            with contextlib.suppress(EOFError):
                line = (await asyncio.to_thread(input, "> ")).strip()
                if line in ("exit", "quit"):
                    self.stop()
                    return
                if line:
                    await self.dialog.handle_text(line)
                continue
            self.stop()
            return
