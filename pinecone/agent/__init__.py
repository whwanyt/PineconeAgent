"""AI Agent：LangChain v1 ``create_agent`` + LangGraph（记忆、工具、MCP）。

- :mod:`.model`   ChatOpenAI 工厂（OpenAI 兼容接口）；
- :mod:`.prompts` 语音助手的系统提示词；
- :mod:`.agent`   VoiceAgent：流式按句输出、会话切分、打断后修复历史。
"""

from .agent import AgentContext, VoiceAgent
from .model import create_chat_model

__all__ = ["AgentContext", "VoiceAgent", "create_chat_model"]
