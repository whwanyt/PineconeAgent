"""短期记忆：多轮对话历史。

- 存储：LangGraph ``AsyncSqliteSaver``（``data/agent_history.db``），按 thread_id（会话）保存每一步状态，
  重启后同一会话可以继续；
- 压缩：``SummarizationMiddleware`` 在历史超过 ``summarize_after_tokens`` 时，把较早的消息
  交给模型压缩成一段摘要，只保留最近 ``keep_messages`` 条原文。树莓派上的语音对话
  上下文越短，首句延迟越低、费用越少；
- 清理：启动时删除超过 ``history_days`` 天的旧会话，防止数据库无限增长。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """下面是你（智能音箱）和用户之前的对话。请用中文把它压缩成一段简洁的摘要，供之后的对话参考。
要求：保留用户提出的需求、做出的决定、提到的事实（人名、时间、地点、偏好）和尚未完成的事情；
省略寒暄和已经完成的细节；不超过 200 字；只输出摘要本身。

<对话>
{messages}
</对话>"""


def create_summarization(
    model: BaseChatModel, *, summarize_after_tokens: int, keep_messages: int
) -> SummarizationMiddleware:
    """创建历史摘要中间件。"""
    return SummarizationMiddleware(
        model,
        trigger=("tokens", summarize_after_tokens),
        keep=("messages", keep_messages),
        summary_prompt=SUMMARY_PROMPT,
    )


def thread_started_at(thread_id: str) -> float | None:
    """从 ``voice-<时间戳>`` 形式的会话 id 中取出创建时间。"""
    prefix, _, ts = thread_id.partition("-")
    return float(ts) if prefix == "voice" and ts.isdigit() else None


def prune_history(db_path: Path, keep_days: float) -> int:
    """删除早于 ``keep_days`` 天的会话，返回删除的会话数。

    直接操作 SQLite（checkpointer 打开之前调用），避免为了清理而遍历全部检查点。
    """
    if not db_path.exists() or keep_days <= 0:
        return 0
    cutoff = time.time() - keep_days * 86400
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "checkpoints" not in tables:
            return 0
        threads = [row[0] for row in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")]
        old = [
            t for t in threads if (started := thread_started_at(t)) is not None and started < cutoff
        ]
        for table in ("checkpoints", "writes"):
            if table in tables:
                conn.executemany(f"DELETE FROM {table} WHERE thread_id = ?", [(t,) for t in old])  # noqa: S608
    if old:
        logger.info("清理了 %d 个过期会话", len(old))
    return len(old)
