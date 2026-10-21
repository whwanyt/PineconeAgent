"""长期记忆：跨会话记住用户的偏好和事实。

- 存储：LangGraph ``AsyncSqliteStore``（``data/agent_memory.db``），命名空间 ``("user", "memories")``；
- 写入：AI 通过 ``remember`` 工具主动记录（“我叫小明”“我不喜欢摇滚”“我每天七点起床”），
  ``forget`` 工具删除过时的记忆；
- 读取：每轮对话开始前，取与本句话最相关的若干条 + 最近记录的若干条，注入系统提示词；
  AI 也可以用 ``recall`` 工具主动查询。

相关性用汉字二元组（bigram）重合度计算：家用音箱的记忆通常只有几十条，不需要向量模型，
在树莓派上零额外开销，也不依赖联网的 embedding 接口。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from pinecone.nlu.matcher import normalize

logger = logging.getLogger(__name__)

NAMESPACE = ("user", "memories")
# 读取时最多扫描的记忆条数（家用场景足够，避免无限增长时拖慢每轮对话）
_SCAN_LIMIT = 500


@dataclass
class Memory:
    key: str
    text: str
    created_at: str


def _bigrams(text: str) -> set[str]:
    text = normalize(text).replace(" ", "")
    return {text[i : i + 2] for i in range(len(text) - 1)} or ({text} if text else set())


def relevance(query: str, text: str) -> float:
    """query 与 text 的二元组重合系数（0~1）。

    分母取两者中较小的集合（overlap coefficient），否则用户的长句子和简短的记忆
    （“我叫什么名字来着” vs “用户叫小明”）永远得分很低。
    """
    q, t = _bigrams(query), _bigrams(text)
    return len(q & t) / min(len(q), len(t)) if q and t else 0.0


class RememberArgs(BaseModel):
    content: str = Field(
        description="要记住的一条信息，用第三人称完整描述，例如“用户叫小明”“用户不喜欢摇滚乐”"
    )


class RecallArgs(BaseModel):
    query: str = Field(description="要查询的内容关键词，例如“名字”“喜欢的歌”")


class ForgetArgs(BaseModel):
    content: str = Field(description="要忘掉的信息（与记住时的描述相近即可）")


class LongTermMemory:
    """长期记忆的读写与工具封装。"""

    def __init__(
        self, store: BaseStore, *, inject_relevant: int = 5, inject_recent: int = 5
    ) -> None:
        self._store = store
        self._inject_relevant = inject_relevant
        self._inject_recent = inject_recent

    async def all(self) -> list[Memory]:
        """全部记忆，按记录时间从旧到新排序。"""
        items = await self._store.asearch(NAMESPACE, limit=_SCAN_LIMIT)
        memories = [
            Memory(key=i.key, text=i.value["text"], created_at=i.value.get("created_at", ""))
            for i in items
        ]
        return sorted(memories, key=lambda m: m.created_at)

    async def add(self, text: str) -> bool:
        """记录一条记忆；已有完全相同的内容时不重复记录。返回是否新增。"""
        text = text.strip()
        if not text:
            return False
        if any(normalize(m.text) == normalize(text) for m in await self.all()):
            return False
        value = {"text": text, "created_at": datetime.now().isoformat(timespec="seconds")}
        await self._store.aput(NAMESPACE, uuid.uuid4().hex, value)
        logger.info("记住：%s", text)
        return True

    async def search(self, query: str, limit: int = 5, min_score: float = 0.2) -> list[Memory]:
        """按相关性检索。"""
        scored = [(relevance(query, m.text), m) for m in await self.all()]
        scored = [(s, m) for s, m in scored if s >= min_score]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    async def forget(self, text: str) -> list[str]:
        """删除与描述最相近的记忆（相关度 >= 0.5），返回被删除的内容。"""
        matched = await self.search(text, limit=3, min_score=0.5)
        for m in matched:
            await self._store.adelete(NAMESPACE, m.key)
            logger.info("忘记：%s", m.text)
        return [m.text for m in matched]

    async def for_prompt(self, text: str) -> list[str]:
        """本轮对话要注入提示词的记忆：最相关的几条 + 最近的几条（去重，保持顺序）。"""
        memories = await self.all()
        if len(memories) <= self._inject_relevant + self._inject_recent:
            return [m.text for m in memories]
        picked = await self.search(text, limit=self._inject_relevant)
        result: dict[str, str] = {m.key: m.text for m in picked}
        for m in memories[-self._inject_recent :]:
            result.setdefault(m.key, m.text)
        return list(result.values())

    def tools(self) -> list[BaseTool]:
        """remember / recall / forget 三个工具。"""

        async def remember(content: str) -> str:
            return "已记住" if await self.add(content) else "之前已经记过了"

        async def recall(query: str) -> str:
            found = await self.search(query)
            return "；".join(m.text for m in found) if found else "没有相关记忆"

        async def forget(content: str) -> str:
            removed = await self.forget(content)
            return "已忘记：" + "；".join(removed) if removed else "没有找到这条记忆"

        return [
            StructuredTool.from_function(
                coroutine=remember,
                name="remember",
                description="长期记住关于用户的信息（称呼、喜好、作息、家庭成员等）。用户明确让你记住，或透露了以后有用的个人信息时调用。",
                args_schema=RememberArgs,
            ),
            StructuredTool.from_function(
                coroutine=recall,
                name="recall",
                description="查询之前记住的关于用户的信息。系统提示词里没有、但可能记过的内容可以用它查。",
                args_schema=RecallArgs,
            ),
            StructuredTool.from_function(
                coroutine=forget,
                name="forget",
                description="删除一条过时或用户要求忘掉的记忆。信息变化时（比如换了称呼）先 forget 旧的再 remember 新的。",
                args_schema=ForgetArgs,
            ),
        ]
