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
import functools
import logging
import re
from typing import Any

from langchain_core.tools import BaseTool, ToolException
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


async def _fail_fast(message: Any) -> None:
    """MCP 会话的消息回调：收到传输层异常时立即抛出。

    mcp 库遇到“回复读到一半断开”等传输错误时，只把异常交给这个回调（默认什么都不做），
    等待中的请求不会失败。这里把异常抛出，会话的接收循环随之结束，并在 finally 中
    让所有等待中的请求以 “Connection closed” 失败，工具调用立即返回而不必等到超时。
    """
    if isinstance(message, Exception):
        raise message


class _DropTraceback(logging.Filter):
    """去掉 mcp 库日志里的 traceback，只保留一行消息；原因由本模块的警告日志给出。

    mcp 的接收循环用根 logger 打印 “Unhandled exception in receive loop”，
    所以根 logger 上也要挂，但只处理这一条，不影响其他日志。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "root" or str(record.msg).startswith("Unhandled exception in receive"):
            record.exc_info = None
            record.exc_text = None
        return True


for _name in ("mcp.client.streamable_http", "mcp.client.sse", "mcp.shared.session", "root"):
    logging.getLogger(None if _name == "root" else _name).addFilter(_DropTraceback())


def _with_fail_fast(conn: dict[str, Any]) -> dict[str, Any]:
    """在连接配置的 session_kwargs 中装上 :func:`_fail_fast`（用户自己配置了回调则不覆盖）。"""
    session_kwargs = dict(conn.get("session_kwargs") or {})
    session_kwargs.setdefault("message_handler", _fail_fast)
    return {**conn, "session_kwargs": session_kwargs}


def _root_cause(exc: BaseException) -> str:
    """取出异常组里的第一个底层异常，生成可读的错误描述。

    MCP 客户端基于 anyio TaskGroup，网络错误会被包成
    “unhandled errors in a TaskGroup”，直接打印看不出原因（如 401、410、连接被拒）。
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def guard_tool(tool: BaseTool, timeout_s: float) -> BaseTool:
    """给 MCP 工具加调用超时，并把连接类错误转成给模型看的文字结果。

    为什么需要：mcp 库在 HTTP 回复读到一半断开时只记录 ERROR 日志、把异常交给
    消息回调，**不会让正在等待的请求失败**，工具调用会一直挂起，整轮对话卡死，
    只能靠唤醒词打断。这里给每次调用加 ``asyncio.timeout``，超时或出错时返回
    “工具暂时不可用”的说明，模型据此正常回复用户（如“联网搜索暂时用不了”）。

    注意：
    - ``ToolException`` 原样抛出，交给适配器的 ``handle_tool_error``（服务端返回的业务错误）；
    - ``CancelledError`` 不是 ``Exception`` 子类，打断 / “停止”时照常取消；
    - 用 ``functools.wraps`` 保留原协程签名，LangChain 依签名注入 runtime 等参数。

    Args:
        tool: langchain-mcp-adapters 生成的工具（返回 ``(内容, 附件)`` 二元组）。
        timeout_s: 单次调用超时（秒）；<= 0 表示不限时。

    Returns:
        替换了协程的工具副本；没有协程的工具原样返回。
    """
    original = getattr(tool, "coroutine", None)
    if original is None or timeout_s <= 0:
        return tool

    @functools.wraps(original)
    async def guarded(*args: Any, **kwargs: Any) -> Any:
        try:
            async with asyncio.timeout(timeout_s):
                return await original(*args, **kwargs)
        except ToolException:
            raise
        except TimeoutError:
            logger.warning("MCP 工具 %s 调用超时（%.0f 秒）", tool.name, timeout_s)
            reason = f"调用超时（{timeout_s:.0f} 秒）"
        except Exception as exc:
            logger.warning("MCP 工具 %s 调用失败：%s", tool.name, _root_cause(exc))
            reason = "连接中断或服务出错"
        text = f"工具 {tool.name} 暂时不可用：{reason}。请如实告诉用户这项功能暂时用不了，不要编造结果。"
        return text, None

    return tool.model_copy(update={"coroutine": guarded})


async def load_mcp_tools(
    servers: dict[str, dict[str, Any]],
    timeout_s: float = 20.0,
    call_timeout_s: float = 30.0,
) -> list[BaseTool]:
    """加载全部 MCP 服务的工具。

    Args:
        servers: 服务名 → 连接配置（格式同 langchain-mcp-adapters 的 connections）。
        timeout_s: 单个服务的加载超时。
        call_timeout_s: 单次工具调用超时，见 :func:`guard_tool`。

    Returns:
        成功加载的工具列表（失败的服务被跳过）。
    """
    valid = {}
    for name, conn in servers.items():
        if not _NAME.match(name):
            logger.error("MCP 服务名 %r 不合法（只能用英文、数字、下划线、横线），已跳过", name)
            continue
        valid[name] = _with_fail_fast(normalize_connection(conn))
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
        tools.extend(guard_tool(t, call_timeout_s) for t in loaded)
    return tools
