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

# Cursor / ModelScope / Claude Desktop 等平台导出的配置常用的传输方式写法 → 适配器认可的写法
_TRANSPORT_ALIASES = {
    "streamablehttp": "streamable_http",
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
    "http": "streamable_http",
    "sse": "sse",
    "stdio": "stdio",
    "websocket": "websocket",
    "ws": "websocket",
}


def normalize_connection(conn: dict[str, Any]) -> dict[str, Any]:
    """把各平台常见的 MCP 配置写法统一成 langchain-mcp-adapters 的格式。

    适配器只认 ``transport`` 字段，而很多平台复制出来的配置写的是 ``type``，
    甚至两者都不写。这里依次处理：
    1. 没有 ``transport`` 时取 ``type``（并删除 ``type``，适配器不接受多余字段）；
    2. 仍然没有时按字段推断：有 ``command`` → stdio，有 ``url`` → streamable_http；
    3. 把 ``streamableHttp`` 等别名统一成适配器的取值。

    Args:
        conn: 单个服务的原始连接配置。

    Returns:
        新的配置字典（不修改入参）；无法识别的取值原样保留，交给适配器报错。
    """
    result = dict(conn)
    transport = result.pop("type", None)
    transport = result.get("transport") or transport
    if not transport:
        if "command" in result:
            transport = "stdio"
        elif "url" in result:
            transport = "streamable_http"
    if transport:
        result["transport"] = _TRANSPORT_ALIASES.get(str(transport).lower(), transport)
    return result


def _root_cause(exc: BaseException) -> str:
    """取出异常组里的第一个底层异常，生成可读的错误描述。

    MCP 客户端基于 anyio TaskGroup，网络错误会被包成
    “unhandled errors in a TaskGroup”，直接打印看不出原因（如 401、410、连接被拒）。
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


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
        valid[name] = normalize_connection(conn)
    if not valid:
        return []

    client = MultiServerMCPClient(valid, tool_name_prefix=True)  # type: ignore[arg-type]
    tools: list[BaseTool] = []
    for name in valid:
        try:
            async with asyncio.timeout(timeout_s):
                loaded = await client.get_tools(server_name=name)
        except Exception as exc:
            logger.warning("MCP 服务 %s 加载失败，已跳过：%s", name, _root_cause(exc))
            continue
        logger.info("MCP 服务 %s：%s", name, "、".join(t.name for t in loaded) or "无工具")
        tools.extend(loaded)
    return tools
