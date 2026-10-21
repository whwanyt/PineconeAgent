"""自定义任务引擎测试：加载校验、变量替换、各动作执行与取消。"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx
import pytest

from pinecone.core.bus import EventBus
from pinecone.core.cancel import CancelToken
from pinecone.core.task_manager import TaskManager, TaskStatus
from pinecone.tasks import TaskRunner, TaskSpec, load_tasks
from pinecone.tasks.actions import ActionContext, ActionError, create_action
from pinecone.tasks.actions.http import pick


def _ctx(**variables) -> tuple[ActionContext, list[str]]:
    spoken: list[str] = []

    async def say(text: str) -> None:
        spoken.append(text)

    return ActionContext(
        task_name="测试", say=say, token=CancelToken(), variables=variables
    ), spoken


# ---------------------------------------------------------------- 供 python 动作调用的函数


def add(a: int, b: int) -> int:
    return int(a) + int(b)


async def async_echo(text: str) -> str:
    return f"echo:{text}"


def slow_loop(token: CancelToken) -> str:
    for _ in range(200):
        token.raise_if_cancelled()
        time.sleep(0.01)
    return "finished"


# ---------------------------------------------------------------- 加载


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """合法任务加载；未知动作、参数错误、重名、下划线开头的文件被跳过。"""
    monkeypatch.setenv("HA_HOST", "ha.local")
    _write(
        tmp_path / "light.yaml",
        """
name: 开灯
keywords: [开灯]
actions:
  - type: http
    url: "http://${HA_HOST}/on"
""",
    )
    _write(tmp_path / "bad_type.yaml", "name: 坏\nkeywords: [x]\nactions: [{type: nope}]\n")
    _write(tmp_path / "bad_param.yaml", "name: 坏2\nkeywords: [x]\nactions: [{type: sleep}]\n")
    _write(
        tmp_path / "z_dup.yaml", "name: 开灯\nkeywords: [y]\nactions: [{type: speak, text: hi}]\n"
    )
    _write(
        tmp_path / "_off.yaml", "name: 禁用\nkeywords: [z]\nactions: [{type: speak, text: hi}]\n"
    )
    _write(
        tmp_path / "bad_re.yaml", "name: 坏3\npatterns: ['(']\nactions: [{type: speak, text: hi}]\n"
    )

    tasks = load_tasks([tmp_path])
    assert [t.name for t in tasks] == ["开灯"]
    assert tasks[0].actions[0].params["url"] == "http://ha.local/on"


def test_render_and_pick() -> None:
    ctx, _ = _ctx(room="客厅", output="ok")
    assert ctx.render("打开{{room}}的灯，{{ output }}，{{missing}}") == "打开客厅的灯，ok，"
    assert ctx.render({"a": ["{{room}}", 1]}) == {"a": ["客厅", 1]}
    # 单花括号原样保留（shell / JSON 中常见）
    assert ctx.render("awk '{print $1}'") == "awk '{print $1}'"
    assert pick({"data": {"items": [{"t": 21}]}}, "data.items.0.t") == 21


# ---------------------------------------------------------------- 动作


async def test_shell_action() -> None:
    ctx, _ = _ctx(name="世界")
    assert await create_action("shell", {"command": "echo 你好{{name}}"}).run(ctx) == "你好世界"
    with pytest.raises(ActionError):
        await create_action("shell", {"command": "exit 3"}).run(ctx)
    assert await create_action("shell", {"command": "exit 3", "check": False}).run(ctx) == ""


async def test_shell_cancel_kills_process_group(tmp_path: Path) -> None:
    """取消时整个进程组（含管道中的子进程）都被终止。"""
    pid_file = tmp_path / "pid"
    action = create_action(
        "shell", {"command": f"echo $$ > {pid_file}; sleep 30 | cat", "grace_s": 1}
    )
    ctx, _ = _ctx()
    task = asyncio.create_task(action.run(ctx))
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.02)
    pgid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


async def test_http_action(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/fail":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"data": {"temp": 23}})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    ctx, _ = _ctx(room="bedroom")
    action = create_action(
        "http",
        {
            "method": "post",
            "url": "http://ha/api/{{room}}",
            "json": {"entity": "light.{{room}}"},
            "result_path": "data.temp",
        },
    )
    assert await action.run(ctx) == "23"
    assert seen[0].method == "POST" and seen[0].url.path == "/api/bedroom"
    assert b"light.bedroom" in seen[0].content
    with pytest.raises(ActionError):
        await create_action("http", {"url": "http://ha/fail"}).run(ctx)


async def test_python_action() -> None:
    ctx, _ = _ctx(x="2")
    assert (
        await create_action(
            "python", {"callable": "tests.test_tasks:add", "args": {"a": "{{x}}", "b": 3}}
        ).run(ctx)
        == "5"
    )
    assert (
        await create_action(
            "python", {"callable": "tests.test_tasks:async_echo", "args": {"text": "hi"}}
        ).run(ctx)
        == "echo:hi"
    )


async def test_python_action_cancel_via_token() -> None:
    """同步函数在线程中运行，通过注入的 token 感知取消。"""
    tasks = TaskManager(EventBus())
    action = create_action("python", {"callable": "tests.test_tasks:slow_loop"})

    async def run(token: CancelToken) -> str | None:
        ctx = ActionContext(task_name="慢", say=_ctx()[0].say, token=token)
        return await action.run(ctx)

    rt = tasks.spawn(run, name="慢任务")
    await asyncio.sleep(0.05)
    rt.cancel("user")
    assert await rt.wait() == TaskStatus.CANCELLED
    assert rt.token.cancelled


async def test_speak_and_sleep() -> None:
    ctx, spoken = _ctx(output="完成")
    await create_action("speak", {"text": "备份{{output}}"}).run(ctx)
    await create_action("sleep", {"seconds": 0}).run(ctx)
    assert spoken == ["备份完成"]


# ---------------------------------------------------------------- 执行器


async def test_runner_output_chain_and_on_error() -> None:
    spoken: list[str] = []

    async def say(text: str) -> None:
        spoken.append(text)

    runner = TaskRunner(say)
    ok = TaskSpec.model_validate(
        {
            "name": "链式",
            "keywords": ["x"],
            "actions": [
                {"type": "shell", "command": "echo 42"},
                {"type": "speak", "text": "结果{{output}}"},
            ],
        }
    )
    assert await runner.run(ok) == "42"
    assert spoken == ["结果42"]

    bad = TaskSpec.model_validate(
        {
            "name": "失败",
            "keywords": ["x"],
            "actions": [{"type": "shell", "command": "exit 1"}],
            "on_error": [{"type": "speak", "text": "出错了"}],
            "on_cancel": [{"type": "speak", "text": "已停止，原因：{{reason}}"}],
        }
    )
    with pytest.raises(ActionError):
        await runner.run(bad)
    assert spoken[-1] == "出错了"
    await runner.run_cleanup(bad, "timeout")
    assert spoken[-1] == "已停止，原因：超时"
