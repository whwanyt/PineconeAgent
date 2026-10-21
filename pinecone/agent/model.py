"""大模型工厂。

统一使用 ``ChatOpenAI`` + ``base_url``：国内主流大模型（DeepSeek、通义、Kimi、智谱）和本地
Ollama 都提供 OpenAI 兼容接口，切换模型只需要改配置，不需要改代码。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from pinecone.core.config import AgentConfig


def create_chat_model(config: AgentConfig) -> BaseChatModel:
    """根据配置创建聊天模型。

    Raises:
        ValueError: 没有配置 api_key。
    """
    if not config.api_key:
        raise ValueError("没有配置 agent.api_key（或环境变量 OPENAI_API_KEY）")
    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        timeout=config.timeout_s,
        max_retries=config.max_retries,
        streaming=True,
    )
