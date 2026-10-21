"""shell 动作：执行一条 shell 命令。

YAML 示例::

    - type: shell
      command: "rsync -a ~/Pictures/ /mnt/backup/pictures/"
      cwd: ~              # 可选，工作目录
      env: {LANG: C}      # 可选，额外环境变量
      grace_s: 3          # 可选，取消时 SIGTERM 后等待多久再 SIGKILL
      check: true         # 可选，返回码非 0 时视为失败（触发 on_error）

取消如何保证进程真的停下：
    命令用 ``start_new_session=True`` 启动，成为新进程组的组长。``sh -c "a | b"`` 会派生多个
    子进程，只杀 sh 本身会留下孤儿进程继续运行，所以取消时对 **整个进程组** 发送 SIGTERM，
    等待 ``grace_s`` 秒仍未退出再发 SIGKILL。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal

from pydantic import BaseModel, field_validator

from .base import Action, ActionContext, ActionError
from .registry import register_action

# 输出最多保留多少字符（作为 {{output}} 传给下一个动作 / 用于播报）
_MAX_OUTPUT = 2000


@register_action
class ShellAction(Action):
    type = "shell"

    class Params(BaseModel):
        command: str
        cwd: str | None = None
        env: dict[str, str] = {}
        grace_s: float = 3.0
        check: bool = True

        @field_validator("cwd")
        @classmethod
        def _expand_home(cls, value: str | None) -> str | None:
            return os.path.expanduser(value) if value else None

    params: Params

    async def run(self, ctx: ActionContext) -> str | None:
        p = self.params
        command = ctx.render(p.command)
        cwd = p.cwd
        env = {**os.environ, **{k: str(ctx.render(v)) for k, v in p.env.items()}}
        ctx.logger.info("[%s] shell：%s", ctx.task_name, command)

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # 独立进程组，取消时整组终止
        )
        try:
            stdout, _ = await proc.communicate()
        except asyncio.CancelledError:
            await self._terminate(proc, p.grace_s, ctx)
            raise

        output = stdout.decode(errors="replace").strip()[-_MAX_OUTPUT:]
        if p.check and proc.returncode != 0:
            raise ActionError(f"命令退出码 {proc.returncode}：{output[-200:]}")
        return output

    @staticmethod
    async def _terminate(
        proc: asyncio.subprocess.Process, grace_s: float, ctx: ActionContext
    ) -> None:
        """先 SIGTERM 整个进程组，宽限期后 SIGKILL。

        本函数在 CancelledError 的处理中执行，期间的 await 不会再次被取消
        （TaskManager 只发一次 cancel），因此可以安全等待进程退出。
        """
        if proc.returncode is not None:
            return
        pgid = proc.pid  # start_new_session=True 时进程组 id == 组长 pid
        ctx.logger.info("[%s] 终止进程组 %d（SIGTERM）", ctx.task_name, pgid)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), grace_s)
        except TimeoutError:
            ctx.logger.warning(
                "[%s] 进程组 %d 未在 %.0fs 内退出，SIGKILL", ctx.task_name, pgid, grace_s
            )
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            await proc.wait()
