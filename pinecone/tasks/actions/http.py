"""http 动作：发送 HTTP 请求（例如调用 Home Assistant、IFTTT、自建服务）。

YAML 示例::

    - type: http
      method: POST                      # 默认 GET
      url: "http://homeassistant.local:8123/api/services/light/turn_on"
      headers: {Authorization: "Bearer ${HA_TOKEN}"}   # ${ENV} 在加载配置时展开
      json: {entity_id: "light.{{room}}"}             # 或 data: "原始文本"
      timeout_s: 10
      result_path: "data.temperature"   # 可选：从 JSON 响应中取值作为 {{output}}

取消：请求通过 httpx 异步发送，任务取消时连接随 CancelledError 一起关闭。
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from .base import Action, ActionContext, ActionError
from .registry import register_action


def pick(data: Any, path: str) -> Any:
    """按点号路径从 JSON 中取值，例如 ``pick(d, "data.items.0.name")``。"""
    for key in path.split("."):
        if isinstance(data, list) and key.isdigit():
            data = data[int(key)] if int(key) < len(data) else None
        elif isinstance(data, dict):
            data = data.get(key)
        else:
            return None
    return data


@register_action
class HttpAction(Action):
    type = "http"

    class Params(BaseModel):
        url: str
        method: str = "GET"
        headers: dict[str, str] = {}
        params: dict[str, Any] = {}
        json_body: Any = Field(default=None, alias="json")
        data: str | None = None
        timeout_s: float = 10.0
        result_path: str | None = None

    params: Params

    async def run(self, ctx: ActionContext) -> str | None:
        p = self.params
        url = ctx.render(p.url)
        ctx.logger.info("[%s] http %s %s", ctx.task_name, p.method.upper(), url)
        async with httpx.AsyncClient(timeout=p.timeout_s) as client:
            resp = await client.request(
                p.method.upper(),
                url,
                headers=ctx.render(p.headers),
                params=ctx.render(p.params),
                json=ctx.render(p.json_body) if p.json_body is not None else None,
                content=ctx.render(p.data) if p.data is not None else None,
            )
        if resp.status_code >= 400:
            raise ActionError(f"HTTP {resp.status_code}：{resp.text[:200]}")
        if p.result_path:
            try:
                value = pick(resp.json(), p.result_path)
            except ValueError as exc:
                raise ActionError("响应不是 JSON，无法按 result_path 取值") from exc
            return "" if value is None else str(value)
        return resp.text
