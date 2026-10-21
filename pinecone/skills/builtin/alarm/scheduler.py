"""闹钟调度（APScheduler）。

- 一次性闹钟 → ``DateTrigger``；重复闹钟 → ``CronTrigger(day_of_week=...)``；
- 调度器只保存在内存中，启动时从 :class:`AlarmStore` 重新排程；
- 关机期间错过的一次性闹钟：错过不超过 ``MISSED_GRACE`` 的立即补响，更早的直接删除。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from .store import Alarm, AlarmStore

logger = logging.getLogger(__name__)

# 关机期间错过的闹钟，在多长时间内开机仍然补响
MISSED_GRACE = timedelta(minutes=10)
_DOW = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

FireCallback = Callable[[Alarm], Awaitable[None]]


class AlarmScheduler:
    """把 AlarmStore 中的闹钟排进 APScheduler。"""

    def __init__(self, store: AlarmStore, on_fire: FireCallback) -> None:
        self._store = store
        self._on_fire = on_fire
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        """启动调度器并加载全部闹钟（必须在事件循环中调用）。"""
        self._scheduler.start()
        now = datetime.now()
        for alarm in self._store.all():
            if alarm.at is not None and alarm.at <= now:
                if now - alarm.at <= MISSED_GRACE:
                    logger.info("补响关机期间错过的闹钟 #%d", alarm.id)
                    self._schedule(alarm, run_at=now + timedelta(seconds=3))
                else:
                    logger.info("删除已过期的闹钟 #%d（%s）", alarm.id, alarm.at)
                    self._store.delete(alarm.id)
                continue
            self._schedule(alarm)

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def add(self, alarm: Alarm) -> None:
        """排程一个新闹钟。"""
        self._schedule(alarm)

    def remove(self, alarm_id: int) -> None:
        """取消排程（闹钟不存在时忽略）。"""
        job = self._scheduler.get_job(str(alarm_id))
        if job is not None:
            job.remove()

    def next_fire_time(self, alarm_id: int) -> datetime | None:
        """下次触发时间（本地时区，去掉时区信息便于比较和播报）。"""
        job = self._scheduler.get_job(str(alarm_id))
        if job is None or job.next_run_time is None:
            return None
        return job.next_run_time.replace(tzinfo=None)

    def _schedule(self, alarm: Alarm, run_at: datetime | None = None) -> None:
        if alarm.weekdays is not None:
            trigger = CronTrigger(
                hour=alarm.hour,
                minute=alarm.minute,
                day_of_week=",".join(_DOW[d] for d in alarm.weekdays),
            )
        else:
            trigger = DateTrigger(run_date=run_at or alarm.at)
        self._scheduler.add_job(
            self._fire,
            trigger,
            args=[alarm.id],
            id=str(alarm.id),
            replace_existing=True,
            misfire_grace_time=int(MISSED_GRACE.total_seconds()),
        )

    async def _fire(self, alarm_id: int) -> None:
        alarm = next((a for a in self._store.all() if a.id == alarm_id), None)
        if alarm is None:
            return
        if not alarm.repeat:
            # 一次性闹钟响过即删除
            self._store.delete(alarm.id)
        await self._on_fire(alarm)
