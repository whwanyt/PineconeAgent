"""任务控制技能。

支持的说法（归一化后匹配）：
    - 停止当前：“停止”“停下”“别说了”“闭嘴”“安静”“算了” → 停止播报和所有前台任务（含闹钟响铃），
      后台任务（音乐、长时间自定义任务）继续；
    - 取消指定：“取消备份”“停止下载任务”“关掉倒计时” → 按任务名模糊匹配并取消；
    - 停止全部：“停止所有任务”“取消全部任务” → 取消所有可取消的任务；
    - 查询：“现在在做什么”“有哪些任务在运行”。

“取消XX”与其他技能的冲突处理：
    “停止播放音乐”“取消明天的闹钟”这类话也以“停止/取消”开头。本技能优先级较高（90），
    但只有当“XX”确实能匹配到 **正在运行的任务** 时才认领，否则让给音乐、闹钟等技能处理。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pinecone.core.task_manager import RunningTask
from pinecone.skills import Intent, Skill, SkillContext, SkillResult, ToolSpec, register_skill


class CancelTaskArgs(BaseModel):
    """AI 调用取消任务工具的参数。"""

    name: str = Field(
        description="要取消的任务名称或关键词，例如“备份”“倒计时”；取消全部时填“全部”"
    )


def _describe(tasks: list[RunningTask]) -> str:
    """把任务列表描述成一句话。"""
    parts = []
    for t in tasks:
        minutes = int(t.elapsed // 60)
        parts.append(f"{t.name}（已运行{minutes}分钟）" if minutes else t.name)
    return "、".join(parts)


@register_skill
class TaskControlSkill(Skill):
    name = "task_control"
    description = "停止或取消正在执行的任务"
    priority = 90
    timeout = 10
    patterns = {
        "stop": [
            r"^(停止|停下|停一下|停|别说了|不要说了|闭嘴|安静|安静点|算了|好了好了|退下)(吧)?$"
        ],
        "cancel_all": [r"(停止|取消|结束|终止|关掉|关闭)(所有|全部)的?任务"],
        "list": [
            r"(现在|正在|当前)(在)?(做什么|干什么|干嘛)",
            r"(有)?(哪些|什么)任务(在|正在)?(运行|执行)",
        ],
        "cancel": [r"^(取消|停止|终止|结束|关掉|关闭)(?P<name>.+?)(任务)?$"],
    }

    async def setup(self, ctx: SkillContext) -> None:
        self._ctx = ctx

    def match(self, text: str) -> Intent | None:
        intent = super().match(text)
        # 只有“XX”能匹配到正在运行的任务时才认领“取消XX”，否则让给其他技能
        if (
            intent is not None
            and intent.name == "cancel"
            and not self._ctx.tasks.find(intent.slot("name", ""))
        ):
            return None
        return intent

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        if intent.name == "stop":
            # 停止播报 + 前台任务；不回复，安静地停下来最符合直觉
            await ctx.interrupt("user")
            return SkillResult()

        if intent.name == "cancel_all":
            count = ctx.tasks.cancel_all("user")
            ctx.logger.info("已取消全部任务：%d 个", count)
            return SkillResult(reply="好的，已停止所有任务" if count else "现在没有正在运行的任务")

        if intent.name == "list":
            tasks = ctx.tasks.list(exclude_caller=True)
            if not tasks:
                return SkillResult(reply="现在没有正在运行的任务", data={"tasks": []})
            return SkillResult(
                reply=f"正在运行：{_describe(tasks)}",
                data={"tasks": [{"id": t.id, "name": t.name, "kind": t.kind.value} for t in tasks]},
            )

        # cancel：AI 调用时 name 可能是“全部”
        name = str(intent.slot("name", "")).strip()
        if name in ("全部", "所有", "all"):
            return await self.handle(
                ctx, Intent(skill=self.name, name="cancel_all", source=intent.source)
            )
        cancelled = ctx.tasks.cancel_by_name(name, "user")
        if not cancelled:
            return SkillResult(reply=f"没有找到正在运行的「{name}」任务", data={"cancelled": []})
        names = "、".join(t.name for t in cancelled)
        return SkillResult(
            reply=f"好的，已取消{names}", data={"cancelled": [t.name for t in cancelled]}
        )

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="list_tasks",
                description="列出音箱上正在运行的任务（音乐、倒计时、自定义任务等）",
                intent="list",
            ),
            ToolSpec(
                name="cancel_task",
                description="按名称取消正在运行的任务；不确定名称时先调用 list_tasks",
                intent="cancel",
                args_schema=CancelTaskArgs,
            ),
            ToolSpec(
                name="stop_speaking",
                description="立即停止当前播报和前台任务（例如闹钟响铃），后台任务继续运行",
                intent="stop",
            ),
        ]
