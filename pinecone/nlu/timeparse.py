"""中文时间表达解析与口语化输出。

支持的表达（识别结果可能是汉字数字，也可能已被 ITN 转成阿拉伯数字，两者都支持）：

    相对时间   “10分钟后”“半小时后”“一个半小时以后”“30秒后”
    绝对时间   “7点”“7点半”“七点一刻”“19:30”“下午3点20”“晚上12点”
    日期       “今天/明天/后天/大后天”“周五/星期五/下周一”“10月25号”“25号”
    重复       “每天”“工作日”“每个周末”“每周一”

没有说明上午 / 下午时，取 **最近的未来时间**：
    例如现在 15:00，说“7点叫我” → 19:00；现在 06:00，说“7点叫我” → 07:00。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# =============================================================================
# 汉字数字
# =============================================================================

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_NUMBER = re.compile(r"[零〇一二两三四五六七八九十百千]+")


def cn_to_int(text: str) -> int:
    """汉字数字转整数：``"二十五"`` → 25，``"十"`` → 10，``"一百零五"`` → 105。已是数字时直接转换。"""
    if text.isdigit():
        return int(text)
    total, num = 0, 0
    for ch in text:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (num or 1) * _CN_UNITS[ch]
            num = 0
    return total + num


def to_digits(text: str) -> str:
    """把文本中的汉字数字全部替换为阿拉伯数字（“七点半”→“7点半”，“星期一”→“星期1”）。"""
    return _CN_NUMBER.sub(lambda m: str(cn_to_int(m.group())), text)


# =============================================================================
# 口语化输出
# =============================================================================

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


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


def spoken_clock(hour: int, minute: int) -> str:
    """口语化钟点，例如 ``下午3点20分``、``早上7点半``、``早上8点``。"""
    hour12 = hour % 12 or 12
    if minute == 0:
        tail = ""
    elif minute == 30:
        tail = "半"
    else:
        tail = f"{minute:02d}分" if minute < 10 else f"{minute}分"
    return f"{period_of_day(hour)}{hour12}点{tail}"


def spoken_day(target: date, today: date) -> str:
    """口语化日期：今天 / 明天 / 后天 / 周X（一周内）/ X月X日。"""
    delta = (target - today).days
    if delta in (0, 1, 2):
        return ["今天", "明天", "后天"][delta]
    if 0 < delta < 7:
        return WEEKDAY_NAMES[target.weekday()]
    return f"{target.month}月{target.day}日"


def spoken_weekdays(weekdays: list[int]) -> str:
    """重复周期的口语描述：每天 / 工作日 / 周末 / 每周一、周三。"""
    days = sorted(set(weekdays))
    if days == list(range(7)):
        return "每天"
    if days == list(range(5)):
        return "工作日"
    if days == [5, 6]:
        return "周末"
    return "每" + "、".join(WEEKDAY_NAMES[d] for d in days)


# =============================================================================
# 解析
# =============================================================================


@dataclass
class ParsedTime:
    """解析结果：一次性时间 ``at``，或重复时间 ``hour:minute`` + ``weekdays``。"""

    hour: int
    minute: int
    at: datetime | None = None
    """一次性闹钟的触发时间。"""
    weekdays: list[int] | None = None
    """重复闹钟的星期（0=周一 … 6=周日）。"""

    @property
    def repeat(self) -> bool:
        return self.weekdays is not None


_PM_PERIODS = ("下午", "傍晚", "晚上", "今晚", "明晚", "夜里")
_NIGHT_PERIODS = ("晚上", "今晚", "明晚", "夜里", "半夜")
_RELATIVE = re.compile(
    r"(?P<h>(?:\d+个?半?|半个?)(?:小时|钟头))?"
    r"(?P<m>\d+分钟?)?"
    r"(?P<s>\d+秒钟?)?"
    r"(?:以后|之后|后)"
)
# “一刻 / 三刻”必须排在分钟数字之前尝试，否则“9点1刻”会被当成 9 点 1 分
_CLOCK = re.compile(r"(?P<h>\d{1,2})(?:点|:|时)(?:(?P<q>[13])刻|(?P<half>半)|(?P<m>\d{1,2})分?)?")
_PERIOD = re.compile(r"凌晨|半夜|清晨|早上|早晨|今早|明早|上午|中午|下午|傍晚|晚上|今晚|明晚|夜里")
# 星期写法统一替换成 “周<序号>”，序号 0=周一 … 6=周日。
# 必须在汉字数字转换之前做，否则“周一8点”会被转成“周18点”。
_WEEKDAY_WORD = re.compile(r"(?:周|星期|礼拜)([一二三四五六日天1-7])")
_WEEKDAY_INDEX = (
    {c: i for i, c in enumerate("一二三四五六日")} | {str(i + 1): i for i in range(7)} | {"天": 6}
)


def _mark_weekdays(text: str) -> str:
    return _WEEKDAY_WORD.sub(lambda m: f"周<{_WEEKDAY_INDEX[m.group(1)]}>", text)


def _parse_relative(text: str, now: datetime) -> ParsedTime | None:
    m = _RELATIVE.search(text)
    if not m or not (m.group("h") or m.group("m") or m.group("s")):
        return None
    hours = 0.0
    if h := m.group("h"):
        number = re.match(r"\d+", h)
        hours = (int(number.group()) if number else 0) + (0.5 if "半" in h else 0)
    minutes = int(re.match(r"\d+", m.group("m")).group()) if m.group("m") else 0  # type: ignore[union-attr]
    seconds = int(re.match(r"\d+", m.group("s")).group()) if m.group("s") else 0  # type: ignore[union-attr]
    at = now + timedelta(hours=hours, minutes=minutes, seconds=seconds)
    return ParsedTime(hour=at.hour, minute=at.minute, at=at)


def _parse_weekdays(text: str) -> list[int] | None:
    if re.search(r"每天|每日|天天|每晚|每早", text):
        return list(range(7))
    if re.search(r"工作日|周<0>到周<4>", text):
        return list(range(5))
    if re.search(r"每个?周末", text):
        return [5, 6]
    found = re.findall(r"每个?周<(\d)>", text)
    return sorted({int(c) for c in found}) if found else None


def _parse_day(text: str, today: date) -> date | None:
    """解析日期部分；没有说明日期时返回 None。"""
    for word, offset in (("大后天", 3), ("后天", 2), ("明", 1), ("今", 0)):
        if word in ("明", "今"):
            if re.search(rf"{word}(天|早|晚|儿)", text):
                return today + timedelta(days=offset)
        elif word in text:
            return today + timedelta(days=offset)
    if m := re.search(r"(下个?)?周<(\d)>", text):
        target = int(m.group(2))
        delta = (target - today.weekday()) % 7
        if m.group(1):
            # “下周一”：下一个自然周的周一
            delta = 7 - today.weekday() + target
        return today + timedelta(days=delta)
    if m := re.search(r"(?:(\d{1,2})月)?(\d{1,2})[日号]", text):
        month = int(m.group(1)) if m.group(1) else today.month
        day = int(m.group(2))
        try:
            target = date(today.year, month, day)
        except ValueError:
            return None
        if target < today:
            # 已过去的日期：只说了“号”就是下个月，说了月份就是明年
            if m.group(1):
                target = date(today.year + 1, month, day)
            else:
                next_month = today.month % 12 + 1
                year = today.year + (1 if next_month == 1 else 0)
                target = date(year, next_month, day)
        return target
    return None


def parse_time(text: str, now: datetime | None = None) -> ParsedTime | None:
    """解析文本中的时间表达。

    Args:
        text: 用户文本（可包含其他内容，例如“明天早上7点叫我起床”）。
        now: 当前时间（测试时注入），默认 ``datetime.now()``。

    Returns:
        解析结果；文本中没有可识别的时间时返回 None。
    """
    now = (now or datetime.now()).replace(microsecond=0)
    text = to_digits(_mark_weekdays(text))

    if relative := _parse_relative(text, now):
        return relative

    clock = _CLOCK.search(text)
    if clock is None:
        return None
    hour = int(clock.group("h"))
    if clock.group("m"):
        minute = int(clock.group("m"))
    elif clock.group("half"):
        minute = 30
    elif clock.group("q"):
        minute = 15 * int(clock.group("q"))
    else:
        minute = 0
    if hour > 24 or minute > 59:
        return None

    period_match = _PERIOD.search(text)
    period = period_match.group() if period_match else ""
    extra_day = 0
    if period in _NIGHT_PERIODS and hour == 12:
        hour, extra_day = 0, 1  # “晚上12点”是第二天 0 点
    elif (period in _PM_PERIODS and hour < 12) or (period == "中午" and hour < 6):
        hour += 12
    elif period == "凌晨" and hour == 12:
        hour = 0
    if hour == 24:
        hour, extra_day = 0, 1

    weekdays = _parse_weekdays(text)
    if weekdays is not None:
        return ParsedTime(hour=hour, minute=minute, weekdays=weekdays)

    day = _parse_day(text, now.date())
    if day is not None:
        at = datetime.combine(day + timedelta(days=extra_day), datetime.min.time()).replace(
            hour=hour, minute=minute
        )
        if not period and hour < 12 and at <= now:
            at += timedelta(hours=12)  # “今天7点”而现在已经过了 7 点 → 晚上 7 点
        return ParsedTime(hour=at.hour, minute=at.minute, at=at)

    # 没有日期：取最近的未来时间；没有时段且 <12 点时，同时考虑上午和下午
    base = datetime.combine(now.date() + timedelta(days=extra_day), datetime.min.time())
    candidates = [base.replace(hour=hour, minute=minute)]
    if not period and hour < 12:
        candidates.append(base.replace(hour=hour + 12, minute=minute))
    future = [c for c in candidates if c > now]
    at = min(future) if future else candidates[0] + timedelta(days=1)
    return ParsedTime(hour=at.hour, minute=at.minute, at=at)
