"""Agent 记忆。

- :mod:`.short_term` 短期记忆：会话历史（LangGraph checkpointer，SQLite），过长时自动摘要压缩；
- :mod:`.long_term`  长期记忆：用户偏好与事实（LangGraph Store，SQLite），remember / recall / forget 工具。
"""
