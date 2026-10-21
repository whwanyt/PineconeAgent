"""示例技能：打招呼 + 一个可被取消的后台倒计时。

复制本目录到 ``plugins/hello/`` 后重启即可使用：
    - 说“你好小派” → 回复问候语；
    - 说“倒计时10秒” → 后台倒计时，结束时播报；期间说“取消倒计时”可终止。

修改要点：
    1. 改 ``name``（全局唯一）和 ``description``；
    2. 在 ``patterns`` 中写正则，用命名分组 ``(?P<名字>...)`` 提取槽位；
    3. 在 ``handle`` 中按 ``intent.name`` 分别处理；
    4. 需要让 AI 也能调用时，实现 ``tools()``。
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from pinecone.skills import Intent, Skill, SkillContext, SkillResult, ToolSpec, register_skill


class CountdownArgs(BaseModel):
    """AI 调用倒计时工具的参数。字段描述会提供给 AI。"""

    seconds: int = Field(description="倒计时秒数，1~3600")


@register_skill
class HelloSkill(Skill):
    name = "hello"
    description = "打招呼和倒计时示例"
    # 正则作用于归一化文本（已去掉标点和空格）
    patterns = {
        "greet": [r"^你好"],
        "countdown": [r"倒计时(?P<seconds>\d+)秒"],
    }

    async def setup(self, ctx: SkillContext) -> None:
        # 读取 config.yaml 中 skills.settings.hello 的私有配置
        self.greeting = ctx.settings.get("greeting", "你好呀，有什么可以帮你？")

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        if intent.name == "greet":
            return SkillResult(reply=self.greeting, follow_up=True)

        seconds = int(intent.slot("seconds", 10))

        async def countdown() -> None:
            await asyncio.sleep(seconds)  # 被取消时这里抛出 CancelledError
            await ctx.say(f"{seconds}秒倒计时结束啦")

        async def on_cancel(reason: str) -> None:
            ctx.logger.info("倒计时被取消：%s", reason)

        # 长时间运行的工作必须用 ctx.spawn 放到后台托管，handle 本身要尽快返回
        ctx.spawn(countdown(), name="倒计时", on_cancel=on_cancel)
        return SkillResult(reply=f"好的，开始{seconds}秒倒计时")

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="start_countdown",
                description="开始一个指定秒数的倒计时，结束时语音提醒",
                intent="countdown",
                args_schema=CountdownArgs,
            )
        ]
