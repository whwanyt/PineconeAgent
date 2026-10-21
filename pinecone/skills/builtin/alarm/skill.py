"""闹钟技能。

说法示例：
    设置   “明天早上7点叫我起床”“10分钟后提醒我关火”“每个工作日7点半的闹钟”“设个闹钟”（会追问时间）
    查询   “我有哪些闹钟”“闹钟列表”
    取消   “取消明天7点的闹钟”“删除起床的闹钟”“取消所有闹钟”
    贪睡   响铃后 “再睡5分钟”“过一会儿再叫我”

响铃行为：
    - 响铃是一个 **前台** 托管任务「闹钟响铃」：说“停止”“关闭闹钟”即可停止；
    - 响铃期间说唤醒词也会立即停止响铃（便于接着说“再睡5分钟”）；
    - 最长响 ``ring_seconds`` 秒（默认 60）后自动停止。

私有配置（config.yaml → skills.settings.alarm）::

    ring_seconds: 60      # 最长响铃时长（秒）
    snooze_minutes: 5     # 贪睡默认分钟数
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel, Field

from pinecone.core.task_manager import TaskKind
from pinecone.nlu.timeparse import parse_time, spoken_clock, spoken_day, spoken_weekdays, to_digits
from pinecone.skills import Intent, Skill, SkillContext, SkillResult, ToolSpec, register_skill

from .scheduler import AlarmScheduler
from .store import Alarm, AlarmStore

# 追问“几点”后的等待时长（秒）：期间直接说时间即可完成设置
_AWAIT_TIME_SECONDS = 15
# 闹钟响过多久内可以“再睡一会儿”
_SNOOZE_WINDOW = timedelta(minutes=30)
_LABEL = re.compile(r"(?:提醒我|提醒一下|叫我|喊我|叫醒我)(?P<label>.+)$")


class SetAlarmArgs(BaseModel):
    when: str = Field(description="中文描述的时间，例如“明天早上7点”“10分钟后”“每个工作日7点半”")
    label: str = Field(default="", description="提醒内容，例如“起床”“开会”，可为空")


class CancelAlarmArgs(BaseModel):
    target: str = Field(
        default="", description="要取消的闹钟：时间（如“明天7点”）、提醒内容，或“全部”"
    )


def make_ringtone(sample_rate: int = 16000) -> np.ndarray:
    """生成约 2 秒的铃声：两组“嘀嘀”声 + 间隔，带淡入淡出避免爆音。"""

    def tone(freq: float, seconds: float) -> np.ndarray:
        t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
        wave = 0.35 * np.sin(2 * np.pi * freq * t)
        fade = int(sample_rate * 0.01)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
        return wave.astype(np.float32)

    gap = np.zeros(int(sample_rate * 0.1), dtype=np.float32)
    pair = np.concatenate([tone(988, 0.15), gap, tone(1319, 0.15), gap])
    return np.concatenate([pair, pair, np.zeros(int(sample_rate * 0.9), dtype=np.float32)])


def extract_label(text: str) -> str:
    """从“明天7点提醒我开会”中提取“开会”。"""
    m = _LABEL.search(text)
    if not m:
        return ""
    label = m.group("label").strip("的一下吧啊呀")
    # 去掉标签中残留的时间表达，例如“叫我7点起床” → “起床”
    return re.sub(r"^\d+点(半|\d+分?)?", "", label).strip()


@register_skill
class AlarmSkill(Skill):
    name = "alarm"
    description = "设置、查询、取消闹钟和提醒"
    timeout = 10
    patterns = {
        "list": [
            # 疑问词必须出现，否则“取消所有闹钟”里的“有闹钟”会误命中
            r"(有|设了|设置了)(哪些|什么|几个)的?(闹钟|提醒)",
            r"(查看|看看|查一下|查查)(一下)?(我的)?(闹钟|提醒)",
            r"(闹钟|提醒)(列表|有哪些|有几个|都有什么)",
        ],
        "cancel": [r"(取消|删除|删掉|关掉|关闭|去掉|不要)(?P<target>.*?)的?(闹钟|提醒)"],
        "snooze": [r"^(再睡|贪睡)", r"(稍后|等一?会儿?|过一?会儿?)再?(叫|喊|提醒)我"],
        "set": [r"闹钟", r"(叫|喊)醒?我", r"提醒(我|一下)"],
    }

    async def setup(self, ctx: SkillContext) -> None:
        self._ctx = ctx
        self._ring_seconds = float(ctx.settings.get("ring_seconds", 60))
        self._snooze_minutes = int(ctx.settings.get("snooze_minutes", 5))
        self._ringtone = make_ringtone()
        self._store = AlarmStore(ctx.data_dir / "alarms.db")
        self._scheduler = AlarmScheduler(self._store, self._on_fire)
        self._scheduler.start()
        self._await_time_until: datetime | None = None
        self._last_ring: tuple[Alarm, datetime] | None = None

    async def teardown(self) -> None:
        self._scheduler.shutdown()
        self._store.close()

    # ------------------------------------------------------------------ 匹配

    def match(self, text: str) -> Intent | None:
        intent = super().match(text)
        if intent is not None:
            return intent
        # 刚追问过“几点”：这句话里只要有时间就当作设置闹钟
        if self._await_time_until and datetime.now() < self._await_time_until and parse_time(text):
            self._await_time_until = None
            return Intent(skill=self.name, name="set", text=text)
        return None

    # ------------------------------------------------------------------ 处理

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        handler = {
            "set": self._set,
            "list": self._list,
            "cancel": self._cancel,
            "snooze": self._snooze,
        }[intent.name]
        return await handler(intent)

    async def _set(self, intent: Intent) -> SkillResult:
        text = str(intent.slot("when") or intent.text)
        now = datetime.now()
        parsed = parse_time(text, now)
        if parsed is None:
            self._await_time_until = now + timedelta(seconds=_AWAIT_TIME_SECONDS)
            return SkillResult(reply="好的，要设置几点的闹钟？", follow_up=True)

        label = str(intent.slot("label") or extract_label(intent.text or text))
        alarm = self._store.add(
            label=label,
            hour=parsed.hour,
            minute=parsed.minute,
            at=parsed.at,
            weekdays=parsed.weekdays,
        )
        self._scheduler.add(alarm)
        self._ctx.logger.info("新增闹钟 #%d：%s %s", alarm.id, self.describe(alarm), label)

        if parsed.at is not None and parsed.at - now < timedelta(hours=1):
            minutes = max(1, round((parsed.at - now).total_seconds() / 60))
            when = f"{minutes}分钟后"
        else:
            when = self.describe(alarm)
        reply = f"好的，{when}提醒你{label}" if label else f"好的，已设置{when}的闹钟"
        return SkillResult(
            reply=reply, data={"id": alarm.id, "when": self.describe(alarm), "label": label}
        )

    async def _list(self, intent: Intent) -> SkillResult:
        alarms = self._sorted_alarms()
        if not alarms:
            return SkillResult(reply="你还没有设置闹钟", data={"alarms": []})
        items = [self.describe(a) + (f"，{a.label}" if a.label else "") for a in alarms]
        return SkillResult(
            reply=f"你有{len(alarms)}个闹钟：" + "；".join(items),
            data={
                "alarms": [{"id": a.id, "when": self.describe(a), "label": a.label} for a in alarms]
            },
        )

    async def _cancel(self, intent: Intent) -> SkillResult:
        alarms = self._sorted_alarms()
        if not alarms:
            return SkillResult(reply="你还没有设置闹钟")
        target = str(intent.slot("target", "")).strip("的 ")

        if target in ("所有", "全部", "所有的", "全部的"):
            matched = alarms
        elif target:
            matched = self._find(alarms, target)
            if not matched:
                return SkillResult(reply=f"没有找到{target}的闹钟")
        elif len(alarms) == 1:
            matched = alarms
        else:
            listing = "；".join(self.describe(a) for a in alarms)
            return SkillResult(
                reply=f"你有{len(alarms)}个闹钟：{listing}。要取消哪一个？", follow_up=True
            )

        for alarm in matched:
            self._scheduler.remove(alarm.id)
            self._store.delete(alarm.id)
        if len(matched) == 1:
            return SkillResult(reply=f"好的，已取消{self.describe(matched[0])}的闹钟")
        return SkillResult(reply=f"好的，已取消{len(matched)}个闹钟")

    async def _snooze(self, intent: Intent) -> SkillResult:
        if self._last_ring is None or datetime.now() - self._last_ring[1] > _SNOOZE_WINDOW:
            return SkillResult(reply="最近没有响过的闹钟")
        # “再睡五分钟” → 5；没说分钟数时用默认值
        number = re.search(r"(\d+)分钟", to_digits(intent.text or ""))
        minutes = int(number.group(1)) if number else self._snooze_minutes
        alarm, _ = self._last_ring
        at = datetime.now().replace(microsecond=0) + timedelta(minutes=minutes)
        snoozed = self._store.add(
            label=alarm.label, hour=at.hour, minute=at.minute, at=at, weekdays=None
        )
        self._scheduler.add(snoozed)
        self._last_ring = None
        return SkillResult(reply=f"好的，{minutes}分钟后再叫你")

    # ------------------------------------------------------------------ 响铃

    async def _on_fire(self, alarm: Alarm) -> None:
        """调度器回调：启动前台托管任务「闹钟响铃」。"""
        self._last_ring = (alarm, datetime.now())
        self._ctx.spawn(
            self._ring(alarm),
            name="闹钟响铃",
            kind=TaskKind.FOREGROUND,
            timeout=self._ring_seconds,
        )

    async def _ring(self, alarm: Alarm) -> None:
        ctx = self._ctx
        me = ctx.tasks.current()

        def stop_on_wake(old: str, new: str) -> None:
            # 响铃期间被唤醒：停止响铃，让用户可以接着说“再睡5分钟”
            if new == "listening" and me is not None:
                me.cancel("wake")

        unsubscribe = ctx.on_state_change(stop_on_wake)
        now = datetime.now()
        text = (
            f"{alarm.label}的时间到了"
            if alarm.label
            else f"闹钟响了，现在是{spoken_clock(now.hour, now.minute)}"
        )
        try:
            rounds = 0
            while True:
                if rounds % 4 == 0:
                    await ctx.say(text)
                await ctx.play_sound(self._ringtone, 16000)
                rounds += 1
        finally:
            unsubscribe()

    # ------------------------------------------------------------------ 工具

    def describe(self, alarm: Alarm) -> str:
        """闹钟的口语描述：“明天早上7点”“工作日早上7点半”。"""
        clock = spoken_clock(alarm.hour, alarm.minute)
        if alarm.weekdays is not None:
            return spoken_weekdays(alarm.weekdays) + clock
        assert alarm.at is not None
        return spoken_day(alarm.at.date(), datetime.now().date()) + clock

    def _sorted_alarms(self) -> list[Alarm]:
        """按下次触发时间排序的闹钟列表。"""
        far = datetime.max
        return sorted(self._store.all(), key=lambda a: self._scheduler.next_fire_time(a.id) or far)

    def _find(self, alarms: list[Alarm], target: str) -> list[Alarm]:
        """按时间或提醒内容查找闹钟。"""
        parsed = parse_time(target)
        if parsed is not None:
            found = [a for a in alarms if (a.hour, a.minute) == (parsed.hour, parsed.minute)]
            # 说了具体日期时，一次性闹钟还要匹配日期
            if re.search(r"今|明|后天|周|星期|号|日", target):
                found = [
                    a
                    for a in found
                    if a.repeat or (a.at and parsed.at and a.at.date() == parsed.at.date())
                ]
            if found:
                return found
        return [a for a in alarms if a.label and (a.label in target or target in a.label)]

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="set_alarm",
                description="设置闹钟或定时提醒",
                intent="set",
                args_schema=SetAlarmArgs,
            ),
            ToolSpec(name="list_alarms", description="查询已设置的全部闹钟", intent="list"),
            ToolSpec(
                name="cancel_alarm",
                description="取消闹钟；不确定时先调用 list_alarms",
                intent="cancel",
                args_schema=CancelAlarmArgs,
            ),
        ]
