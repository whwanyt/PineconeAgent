"""自定义任务技能。

路由：
    优先级 100（最高）：用户自己配置的关键词最明确，应该先于内置技能匹配。
    关键词按“包含”匹配（归一化后），多个任务都命中时取匹配最长的那个；
    正则的命名分组会作为 ``{{变量}}`` 传给动作。

执行：
    每次触发都会启动一个以 **任务名** 命名的托管任务（来源 ``task:<任务名>``），因此可以说
    “取消备份”“现在在做什么”。
    - 前台任务（默认）：等待执行完成。动作里一般自带 speak 播报结果；失败时播报“XX执行失败”。
      说“停止”或唤醒打断会取消它；
    - 后台任务（``background: true``）：立即回复 ``reply``（默认“好的，开始XX”），完成后播报
      ``done_reply``，失败时播报“XX执行失败”。同名任务正在运行时不会重复启动；
    - 超时或被取消后执行 ``on_cancel`` 动作，变量 ``{{reason}}`` 为取消原因。

AI：
    ``expose_to_agent: true`` 的任务各自生成一个工具 ``run_task_<序号>``，说明来自任务的 description。

私有配置（config.yaml → skills.settings.custom_task）::

    dirs: [tasks]     # 任务目录（相对项目根目录）
"""

from __future__ import annotations

import re
from typing import Any

from pinecone.core.task_manager import TaskKind, TaskStatus
from pinecone.nlu.matcher import normalize
from pinecone.skills import Intent, Skill, SkillContext, SkillResult, ToolSpec, register_skill
from pinecone.tasks import TaskRunner, TaskSpec, load_tasks
from pinecone.tasks.actions import render

# 关键词前面紧跟这些词时不触发：“取消备份”应该交给 task_control 去取消，而不是再启动一次备份
_NEGATED = re.compile(r"(取消|停止|终止|结束|不要|别|不用)$")


@register_skill
class CustomTaskSkill(Skill):
    name = "custom_task"
    description = "执行用户在 tasks/ 目录中自定义的任务"
    priority = 100
    # 前台任务在 handle 中等待执行完成，超时由每个任务自己的 timeout 控制
    timeout = None

    async def setup(self, ctx: SkillContext) -> None:
        self._ctx = ctx
        dirs = [ctx.config.resolve_path(d) for d in ctx.settings.get("dirs", ["tasks"])]
        self._runner = TaskRunner(ctx.say)
        self._tasks: dict[str, TaskSpec] = {}
        self.reload(load_tasks(dirs))
        ctx.extras["custom_tasks"] = self

    def reload(self, tasks: list[TaskSpec]) -> None:
        """替换任务列表（测试或热重载用）。"""
        self._tasks = {t.name: t for t in tasks}
        self._keywords = [(normalize(k), t.name) for t in tasks for k in t.keywords]
        self._patterns = [(re.compile(p), t.name) for t in tasks for p in t.patterns]

    @property
    def tasks(self) -> list[TaskSpec]:
        return list(self._tasks.values())

    # ------------------------------------------------------------------ 匹配

    def match(self, text: str) -> Intent | None:
        # 正则优先（更精确），然后是关键词，关键词取匹配最长的
        for pattern, task_name in self._patterns:
            if m := pattern.search(text):
                slots = {k: v for k, v in m.groupdict().items() if v is not None}
                return Intent(
                    skill=self.name, name="run", text=text, slots={**slots, "task": task_name}
                )
        hits = [
            (len(k), name)
            for k, name in self._keywords
            if k and k in text and not _NEGATED.search(text[: text.find(k)])
        ]
        if hits:
            _, task_name = max(hits)
            return Intent(skill=self.name, name="run", text=text, slots={"task": task_name})
        return None

    # ------------------------------------------------------------------ 执行

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        spec = self._tasks.get(str(intent.slot("task", "")))
        if spec is None:
            return SkillResult(reply="没有找到这个自定义任务")
        variables: dict[str, Any] = {k: v for k, v in intent.slots.items() if k != "task"}
        variables["text"] = intent.text

        if spec.background:
            return self._start_background(spec, variables)

        rt = await ctx.tasks.run(
            lambda token: self._runner.run(spec, variables),
            **self._task_options(spec, variables, TaskKind.FOREGROUND),
        )
        if rt.status == TaskStatus.DONE:
            return SkillResult(
                reply=self._render(spec.reply, variables, rt.result), data={"output": rt.result}
            )
        if rt.status == TaskStatus.FAILED:
            # 配置了 on_error 时它已经播报过失败原因
            reply = None if spec.on_error else f"{spec.name}执行失败"
            return SkillResult(reply=reply, data={"error": rt.error})
        if rt.status == TaskStatus.TIMEOUT and not spec.on_cancel:
            return SkillResult(reply=f"{spec.name}超时了")
        # 被取消：on_cancel 已经处理过播报，这里保持安静
        return SkillResult(data={"status": rt.status.value})

    def _start_background(self, spec: TaskSpec, variables: dict[str, Any]) -> SkillResult:
        if any(t.name == spec.name for t in self._ctx.tasks.list(TaskKind.BACKGROUND)):
            return SkillResult(reply=f"{spec.name}正在进行中")

        async def run_and_report() -> str | None:
            try:
                output = await self._runner.run(spec, variables)
            except Exception:
                if not spec.on_error:
                    await self._ctx.say(f"{spec.name}执行失败")
                raise
            if spec.done_reply:
                await self._ctx.say(self._render(spec.done_reply, variables, output) or "")
            return output

        self._ctx.tasks.spawn(
            run_and_report(), **self._task_options(spec, variables, TaskKind.BACKGROUND)
        )
        reply = (
            self._render(spec.reply, variables, None) if spec.reply else f"好的，开始{spec.name}"
        )
        return SkillResult(reply=reply)

    def _task_options(
        self, spec: TaskSpec, variables: dict[str, Any], kind: TaskKind
    ) -> dict[str, Any]:
        async def on_cancel(reason: str) -> None:
            await self._runner.run_cleanup(spec, reason, variables)

        return {
            "name": spec.name,
            "kind": kind,
            "source": f"task:{spec.name}",
            "timeout": spec.timeout,
            "cancellable": spec.cancellable,
            "on_cancel": on_cancel,
        }

    @staticmethod
    def _render(template: str | None, variables: dict[str, Any], output: Any) -> str | None:
        if not template:
            return None
        return render(template, {**variables, "output": "" if output is None else output})

    # ------------------------------------------------------------------ AI 工具

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=f"run_task_{i}",
                description=f"执行自定义任务「{spec.name}」"
                + (f"：{spec.description}" if spec.description else ""),
                intent="run",
                fixed_slots={"task": spec.name},
            )
            for i, spec in enumerate(self._tasks.values(), 1)
            if spec.expose_to_agent
        ]
