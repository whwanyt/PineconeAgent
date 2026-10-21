"""中文时间解析测试。"""

from datetime import datetime

import pytest

from pinecone.nlu.timeparse import cn_to_int, parse_time, spoken_clock, to_digits

# 2026-10-21 是星期三，下午 3 点
NOW = datetime(2026, 10, 21, 15, 0)


def test_chinese_numbers() -> None:
    """汉字数字转换。"""
    assert [cn_to_int(s) for s in ("十", "二十五", "一百零五", "两")] == [10, 25, 105, 2]
    assert to_digits("明天早上七点半") == "明天早上7点半"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("十分钟后提醒我", datetime(2026, 10, 21, 15, 10)),
        ("一个半小时以后叫我", datetime(2026, 10, 21, 16, 30)),
        ("明天早上七点半叫我起床", datetime(2026, 10, 22, 7, 30)),
        ("7点叫我", datetime(2026, 10, 21, 19, 0)),  # 没说上下午 → 最近的未来
        ("下午3点20提醒我开会", datetime(2026, 10, 21, 15, 20)),
        ("下午2点", datetime(2026, 10, 22, 14, 0)),  # 今天已过 → 明天
        ("晚上12点", datetime(2026, 10, 22, 0, 0)),
        ("中午1点", datetime(2026, 10, 22, 13, 0)),  # 今天 13 点已过 → 明天
        ("周五上午九点一刻", datetime(2026, 10, 23, 9, 15)),
        ("下周一8点", datetime(2026, 10, 26, 8, 0)),
        ("25号早上6点", datetime(2026, 10, 25, 6, 0)),
        ("19:45", datetime(2026, 10, 21, 19, 45)),
    ],
)
def test_parse_once(text: str, expected: datetime) -> None:
    """一次性时间解析。"""
    parsed = parse_time(text, NOW)
    assert parsed is not None and parsed.at == expected


def test_parse_repeat() -> None:
    """重复时间解析：每天、工作日、每周X。"""
    daily = parse_time("每天早上7点叫我", NOW)
    assert daily and daily.weekdays == list(range(7)) and (daily.hour, daily.minute) == (7, 0)
    workday = parse_time("工作日七点半", NOW)
    assert workday and workday.weekdays == [0, 1, 2, 3, 4] and workday.minute == 30
    weekly = parse_time("每周一晚上8点", NOW)
    assert weekly and weekly.weekdays == [0] and weekly.hour == 20


def test_no_time() -> None:
    """没有时间表达时返回 None。"""
    assert parse_time("帮我设个闹钟", NOW) is None


def test_spoken_clock() -> None:
    """口语化钟点。"""
    assert spoken_clock(7, 30) == "早上7点半"
    assert spoken_clock(15, 5) == "下午3点05分"
