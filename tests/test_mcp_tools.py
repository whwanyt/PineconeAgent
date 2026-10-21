"""MCP 工具接入测试：用本地 stdio 测试服务，验证加载、调用、坏服务跳过。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import StructuredTool

from pinecone.agent.tools.mcp_tools import (
    _fail_fast,
    _with_fail_fast,
    guard_tool,
    load_mcp_tools,
    normalize_connection,
)

SERVER = str(Path(__file__).with_name("mcp_demo_server.py"))


async def test_load_and_call() -> None:
    tools = await load_mcp_tools(
        {
            "demo": {"transport": "stdio", "command": sys.executable, "args": [SERVER]},
            "broken": {"transport": "stdio", "command": "/nonexistent/mcp-server", "args": []},
            "坏名字": {"transport": "stdio", "command": sys.executable, "args": [SERVER]},
        },
        timeout_s=30,
    )
    by_name = {t.name: t for t in tools}
    # 只加载了合法且能连上的服务，工具名带服务名前缀
    assert set(by_name) == {"demo_add", "demo_weather"}
    assert "3" in str(await by_name["demo_add"].ainvoke({"a": 1, "b": 2}))
    assert "北京晴" in str(await by_name["demo_weather"].ainvoke({"city": "北京"}))


async def test_no_servers() -> None:
    """没有配置任何服务时返回空列表。"""
    assert await load_mcp_tools({}) == []


def test_normalize_connection() -> None:
    """兼容各平台的写法：type 代替 transport、驼峰别名、不写类型时按字段推断。"""
    url = "https://example.com/mcp"
    assert normalize_connection({"type": "streamable_http", "url": url}) == {
        "transport": "streamable_http",
        "url": url,
    }
    assert normalize_connection({"type": "streamableHttp", "url": url})["transport"] == (
        "streamable_http"
    )
    assert normalize_connection({"type": "sse", "url": url})["transport"] == "sse"
    assert normalize_connection({"url": url})["transport"] == "streamable_http"
    assert normalize_connection({"command": "uvx", "args": []})["transport"] == "stdio"
    # 已经写了 transport 的以 transport 为准，多余的 type 被删除
    assert normalize_connection({"transport": "stdio", "type": "sse", "command": "x"}) == {
        "transport": "stdio",
        "command": "x",
    }


def _fake_mcp_tool(coroutine: Any) -> StructuredTool:
    """构造与适配器同形的工具：返回 (内容, 附件) 二元组。"""
    return StructuredTool.from_function(
        coroutine=coroutine,
        name="svc_search",
        description="搜索",
        response_format="content_and_artifact",
    )


async def test_guard_timeout() -> None:
    """工具挂起（mcp 库断线后不返回）时按超时返回说明文字，而不是一直等待。"""

    async def hang(query: str) -> tuple[str, None]:
        await asyncio.sleep(60)
        return "不会返回", None

    tool = guard_tool(_fake_mcp_tool(hang), timeout_s=0.1)
    async with asyncio.timeout(5):
        result = await tool.ainvoke({"query": "新闻"})
    assert "暂时不可用" in result and "超时" in result


async def test_guard_error() -> None:
    """连接中断等异常转成说明文字；正常调用结果不受影响。"""

    async def broken(query: str) -> tuple[str, None]:
        raise ConnectionError("peer closed connection")

    async def ok(query: str) -> tuple[str, None]:
        return f"结果：{query}", None

    assert "连接中断" in await guard_tool(_fake_mcp_tool(broken), 5).ainvoke({"query": "x"})
    assert await guard_tool(_fake_mcp_tool(ok), 5).ainvoke({"query": "x"}) == "结果：x"


async def test_guard_keeps_cancellation() -> None:
    """打断时取消照常传播，不会被转成文字结果吞掉。"""

    async def hang(query: str) -> tuple[str, None]:
        await asyncio.sleep(60)
        return "", None

    task = asyncio.ensure_future(guard_tool(_fake_mcp_tool(hang), 30).ainvoke({"query": "x"}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_fail_fast_handler() -> None:
    """传输层异常由消息回调立即抛出（让等待中的请求马上失败）；用户自己的回调不被覆盖。"""
    with pytest.raises(ConnectionError):
        await _fail_fast(ConnectionError("peer closed connection"))
    await _fail_fast(object())  # 普通消息 / 通知不受影响

    conn = _with_fail_fast({"transport": "stdio", "command": "x"})
    assert conn["session_kwargs"]["message_handler"] is _fail_fast

    async def mine(message: Any) -> None:
        return None

    custom = _with_fail_fast({"transport": "stdio", "session_kwargs": {"message_handler": mine}})
    assert custom["session_kwargs"]["message_handler"] is mine


async def test_load_with_type_key() -> None:
    """用 type 字段（Cursor / ModelScope 风格）配置的服务也能正常加载。"""
    tools = await load_mcp_tools(
        {"demo": {"type": "stdio", "command": sys.executable, "args": [SERVER]}},
        timeout_s=30,
    )
    assert {t.name for t in tools} == {"demo_add", "demo_weather"}
