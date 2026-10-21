"""MCP 工具接入（langchain-mcp-adapters）。

MCP（Model Context Protocol）服务把外部能力（天气、搜索、Home Assistant、文件、日历……）
以统一协议暴露成工具。在 config.yaml 的 ``agent.mcp_servers`` 中配置后，启动时自动加载，
AI 就能直接调用，不需要为每个服务写技能。

配置示例::

    agent:
      mcp_servers:
        weather:                       # 服务名（英文、数字、下划线），会作为工具名前缀
          transport: stdio             # 本地进程
          command: uvx
          args: [mcp-weather-server]
        homeassistant:
          transport: streamable_http   # 远程 HTTP 服务
          url: http://homeassistant.local:8123/mcp_server/sse
          headers: {Authorization: "Bearer ${HA_TOKEN}"}

设计要点：
    - 逐个服务加载并设置超时，某个服务连不上只记录警告，不影响其他服务和主程序启动；
    - 工具名加 ``服务名_`` 前缀，避免与技能工具或其他服务重名；
    - 工具调用运行在「对话」托管任务中，打断 / “停止”时随之取消；
      工具内部报错会作为文本结果返回给模型，而不是让整轮对话失败。
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


async def load_mcp_tools(
    servers: dict[str, dict[str, Any]], timeout_s: float = 20.0
) -> list[BaseTool]:
    """加载全部 MCP 服务的工具。

    Args:
        servers: 服务名 → 连接配置（格式同 langchain-mcp-adapters 的 connections）。
        timeout_s: 单个服务的加载超时。

    Returns:
        成功加载的工具列表（失败的服务被跳过）。
    """
    valid = {}
    for name, conn in servers.items():
        if not _NAME.match(name):
            logger.error("MCP 服务名 %r 不合法（只能用英文、数字、下划线、横线），已跳过", name)
            continue
        valid[name] = dict(conn)
    if not valid:
        return []

    client = MultiServerMCPClient(valid, tool_name_prefix=True)  # type: ignore[arg-type]
    tools: list[BaseTool] = []
    for name in valid:
        try:
            async with asyncio.timeout(timeout_s):
                loaded = await client.get_tools(server_name=name)
        except Exception as exc:
            logger.warning("MCP 服务 %s 加载失败，已跳过：%s", name, exc)
            continue
        logger.info("MCP 服务 %s：%s", name, "、".join(t.name for t in loaded) or "无工具")
        tools.extend(loaded)
    return tools
