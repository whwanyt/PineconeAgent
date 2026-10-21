"""MCP 工具接入测试：用本地 stdio 测试服务，验证加载、调用、坏服务跳过。"""

from __future__ import annotations

import sys
from pathlib import Path

from pinecone.agent.tools.mcp_tools import load_mcp_tools, normalize_connection

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


async def test_load_with_type_key() -> None:
    """用 type 字段（Cursor / ModelScope 风格）配置的服务也能正常加载。"""
    tools = await load_mcp_tools(
        {"demo": {"type": "stdio", "command": sys.executable, "args": [SERVER]}},
        timeout_s=30,
    )
    assert {t.name for t in tools} == {"demo_add", "demo_weather"}
