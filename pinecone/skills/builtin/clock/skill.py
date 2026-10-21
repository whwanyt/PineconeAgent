"""时间技能。

说法示例：“现在几点”“几点了”“今天几号”“今天星期几”“今天是什么日子”“明天星期几”。
回复使用口语化的中文时间，例如“现在是下午3点20分”“今天是10月21日，星期三”。

同时暴露 ``get_current_time`` 工具：大模型本身不知道当前时间，
回答“距离周末还有几天”这类问题时需要先调用它。
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pinecone.skills import Intent, Skill, SkillContext, SkillResult, ToolSpec, register_skill

WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def period_of_day(hour: int) -> str:
    """按小时返回口语时段。"""
    if hour < 5:
        return "凌晨"
    if hour < 9:
        return "早上"
    if hour < 12:
        return "上午"
    if hour < 13:
        return "中午"
    if hour < 18:
        return "下午"
    return "晚上"


def spoken_time(dt: datetime) -> str:
    """口语化时间，例如 ``下午3点20分``、``早上8点整``、``中午12点05分``。"""
    hour12 = dt.hour % 12 or 12
    minute = (
        "整" if dt.minute == 0 else f"{dt.minute:02d}分" if dt.minute < 10 else f"{dt.minute}分"
    )
    return f"{period_of_day(dt.hour)}{hour12}点{minute}"


def spoken_date(dt: datetime) -> str:
    """口语化日期，例如 ``10月21日，星期三``。"""
    return f"{dt.month}月{dt.day}日，{WEEKDAYS[dt.weekday()]}"


@register_skill
class ClockSkill(Skill):
    name = "clock"
    description = "查询时间和日期"
    timeout = 5
    patterns = {
        "time": [r"几点", r"(现在|当前)(的)?时间", r"什么时间了"],
        "date": [
            r"(?P<day>今天|明天|后天|昨天)?(是)?(几号|几月几号|星期几|周几|礼拜几|什么日子|日期)",
        ],
    }

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        now = datetime.now()
        if intent.name == "time":
            return SkillResult(
                reply=f"现在是{spoken_time(now)}", data={"now": now.isoformat(timespec="minutes")}
            )

        day = intent.slot("day", "今天")
        offset = {"昨天": -1, "今天": 0, "明天": 1, "后天": 2}.get(day, 0)
        target = now + timedelta(days=offset)
        return SkillResult(
            reply=f"{day}是{spoken_date(target)}",
            data={"date": target.date().isoformat(), "weekday": WEEKDAYS[target.weekday()]},
        )

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="get_current_time",
                description="获取当前的本地日期、星期和时间。回答任何与“现在”“今天”相关的问题前先调用",
                intent="time",
            )
        ]
