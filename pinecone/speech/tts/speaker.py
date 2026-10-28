"""Speaker：面向业务的“说话”接口。

技能、Agent 只需要调用::

    await speaker.say("现在是三点二十分")
    await speaker.say_stream(agent_token_stream)   # 大模型流式输出

内部流水线（合成与播放并行，第 n 句播放时第 n+1 句已经在合成）::

    文本流 ──分句──▶ 句子队列 ──合成──▶ PlaybackHandle.feed ──▶ 扬声器
    （生产者协程）                     （同一个播放句柄，句子之间无缝衔接）

界面：
    传入事件总线时，每句送入扬声器后发布 :class:`~pinecone.core.events.Spoken`，
    文本界面据此显示“正在说什么”。

可取消：
    ``say`` / ``say_stream`` 所在任务被取消（打断、“别说了”）时，合成协程一并取消，
    播放句柄立即停止，扬声器在约 20ms 内静音。
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import AsyncIterable, AsyncIterator
from typing import TYPE_CHECKING

from pinecone.audio.output import AudioOutput
from pinecone.core.events import Spoken
from pinecone.speech.tts.base import TextToSpeech
from pinecone.speech.tts.text import SentenceSplitter, split_sentences

if TYPE_CHECKING:
    from pinecone.core.bus import EventBus

logger = logging.getLogger(__name__)


class Speaker:
    """语音播报器。"""

    def __init__(self, tts: TextToSpeech, output: AudioOutput, bus: EventBus | None = None) -> None:
        self._tts = tts
        self._output = output
        self._bus = bus
        self._utterances = itertools.count(1)

    async def say(self, text: str) -> str:
        """播报一段文本（自动分句流水线合成），播完后返回。

        Returns:
            实际播报的文本。
        """
        return await self._speak(_aiter(split_sentences(text)))

    async def say_stream(self, pieces: AsyncIterable[str]) -> str:
        """播报流式文本（例如大模型逐 token 输出），播完后返回完整文本。"""
        return await self._speak(_split_stream(pieces))

    def stop(self) -> None:
        """立即停止所有播报。"""
        self._output.stop_all()

    async def _speak(self, sentences: AsyncIterator[str]) -> str:
        handle = self._output.open()
        spoken: list[str] = []
        utterance = next(self._utterances)

        async def produce() -> None:
            # 逐句合成并送入播放句柄；单句合成失败时跳过该句，不中断整段播报
            try:
                async for sentence in sentences:
                    try:
                        samples, rate = await self._tts.synthesize(sentence)
                    except Exception:
                        logger.exception("合成失败，跳过：%s", sentence)
                        continue
                    handle.feed(samples, rate)
                    spoken.append(sentence)
                    if self._bus is not None:
                        await self._bus.publish(Spoken(text=sentence, utterance=utterance))
            finally:
                handle.finish()

        producer = asyncio.create_task(produce())
        try:
            await handle.wait()
            await producer  # 传播生产者中的异常
        finally:
            # 被取消或出错时：停止播放、取消合成
            if not producer.done():
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
            handle.stop()
        text = "".join(spoken)
        logger.info("播报完成：%s", text)
        return text


async def _aiter(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


async def _split_stream(pieces: AsyncIterable[str]) -> AsyncIterator[str]:
    """把文本片段流转换成句子流。"""
    splitter = SentenceSplitter()
    async for piece in pieces:
        for sentence in splitter.feed(piece):
            yield sentence
    for sentence in splitter.flush():
        yield sentence
