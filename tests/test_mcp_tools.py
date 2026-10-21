"""MCP 工具接入测试：用本地 stdio 测试服务，验证加载、调用、坏服务跳过。"""

from __future__ import annotations

import sys
from pathlib import Path

from pinecone.agent.tools.mcp_tools import load_mcp_tools

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
    assert await load_mcp_tools({}) == []
