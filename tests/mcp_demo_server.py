"""测试用的最小 MCP 服务（stdio）：提供 add 和 weather 两个工具。"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("demo")


@server.tool()
def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b


@server.tool()
def weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}晴，25度"


if __name__ == "__main__":
    server.run("stdio")
