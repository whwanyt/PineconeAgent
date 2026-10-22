"""天气预报技能（数据来自高德 Web 服务）。

说法示例：
    实况   “现在外面多少度”“现在天气怎么样”
    当天   “今天天气怎么样”“查一下上海的天气”（实况 + 今天的预报）
    预报   “北京明天天气”“后天会下雨吗”“周六杭州天气”“周末天气”
    多天   “未来几天天气”“这几天天气怎么样”
没说城市时依次使用：配置的默认城市 → 出口 IP 所在城市 → 追问“你想查哪个城市”，
追问后的 15 秒内直接说城市名即可。

在架构中的位置：
    与其他技能一样只依赖 ``SkillContext``；HTTP 请求和缓存在同目录的 ``client.py``。
    规则命中与 AI 调用 ``get_weather`` 工具走同一个 :meth:`WeatherSkill.handle`。

私有配置（config.yaml → skills.settings.weather）::

    api_key: ${AMAP_KEY:}   # 高德 Web 服务 Key
    default_city: ""        # 默认城市（名称或 adcode），空则按 IP 定位
    timeout_s: 8            # 单次请求超时（秒）
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from pinecone.skills import Intent, Skill, SkillContext, SkillResult, ToolSpec, register_skill

from .client import AmapClient, AmbiguousPlace, DayForecast, LiveWeather, Place, WeatherError

# 追问城市后的等待时长（秒）：期间直接说城市名即可
_AWAIT_CITY_SECONDS = 15
# 追问时只接受像地名的回答：2~10 个汉字，避免把无关的话当成城市
_CITY_ANSWER = re.compile(r"[\u4e00-\u9fff]{2,10}")

WEEK_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
_RELATIVE_DAYS = {0: "今天", 1: "明天", 2: "后天", 3: "大后天"}
_WEEKDAY_NUM = {
    "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6,
}  # fmt: skip

# 时间表达。“大后天”必须写在“后天”前面，否则会先匹配到“后天”
_WHEN = re.compile(
    r"大后天|后天|明天|明日|今天|今日|今晚|今早|现在|当前|此刻|目前"
    r"|(未来|最近|近|这|接下来)(几|两|三|四|2|3|4)天"
    r"|(下)?(周|星期|礼拜)[一二三四五六日天1-7]|周末"
)
# 说法开头的客套 / 动词，去掉后剩下的就是城市
_FILLER_HEAD = re.compile(
    r"^(那么)?(请问|请|麻烦你?|我想知道|我想问问?|你知道)?(帮我|给我|帮忙)?"
    r"(查询一下|查询|查一下|查查|查|看一下|看看|看|说一下|说说|讲一下|讲讲"
    r"|播报一下|播报|报一下|告诉我|问一下|问问)?(一下)?"
)
# 表示“本地”的词，等于没说城市
_LOCAL_WORDS = re.compile(r"外面|室外|户外|这边|这里|我们这边?|我们这里|咱们这边?|本地|当地")
_RAIN = re.compile(r"雨|雪|带伞|带雨伞")
# 触发词：出现任意一个才算在问天气。不匹配单独的“多少度”“温度”，
# 以免把“空调温度调到26度”“烤箱多少度”当成天气
_TRIGGERS = [
    re.compile(p)
    for p in (
        r"天气",
        r"气温",
        r"(外面|室外|户外)的?温度",
        r"(外面|室外|户外)(多少|几)度",
        r"(会不会|会|要不要|下不下|有没有|可能)(下雨|下雪|有雨|有雪|雨|雪)",
        r"下(雨|雪)吗",
        r"(要不要|用不用|需要|得)带(雨)?伞",
    )
]


class WeatherArgs(BaseModel):
    """AI 调用 ``get_weather`` 的参数。"""

    city: str = Field(
        default="", description="城市或区县名，例如“北京”“杭州西湖区”；为空表示用户所在城市"
    )
    when: str = Field(
        default="今天",
        description="要查的时间：现在、今天、明天、后天、大后天、周六、周末、未来几天",
    )


@dataclass(frozen=True)
class When:
    """解析后的查询时间。"""

    kind: str
    """now：只看实况；today：实况 + 今天预报；dates：指定几天的预报；days：全部预报。"""
    offsets: tuple[int, ...] = ()
    """kind 为 dates 时，相对今天的天数（0 为今天）。"""


def parse_when(text: str, today: date) -> When:
    """把“明天”“周六”“未来几天”等时间说法解析成 :class:`When`。无法识别时按今天处理。"""
    m = _WHEN.search(text or "")
    if m is None:
        return When("today")
    word = m.group(0)
    if word in ("现在", "当前", "此刻", "目前"):
        return When("now")
    if word in ("今天", "今日", "今晚", "今早"):
        return When("today")
    if word in ("明天", "明日"):
        return When("dates", (1,))
    if word == "后天":
        return When("dates", (2,))
    if word == "大后天":
        return When("dates", (3,))
    if word.endswith("天") and not re.search(r"(周|星期|礼拜)天$", word):
        return When("days")
    wd = today.weekday()
    if word == "周末":
        # 今天是周日时，周末只剩今天；周六则是今天和明天
        return When("dates", tuple(sorted({(5 - wd) % 7, (6 - wd) % 7})))
    target = _WEEKDAY_NUM[word[-1]]
    if word.startswith("下"):
        return When("dates", (7 - wd + target,))
    return When("dates", ((target - wd) % 7,))


def parse_query(text: str) -> dict[str, Any]:
    """从归一化文本中抽出城市、时间和是否在问下雨 / 下雪。

    做法：城市一般出现在时间词和“天气”“会不会下雨”之前，
    所以取触发词之前的部分，去掉时间词、客套话、“外面”等，剩下的就是城市。

    Returns:
        槽位字典：``city``（可能为空）、``when``（原始时间词，可能为空）、
        ``rain``（问的是“雨”或“雪”，没问为空）。
    """
    starts = [m.start() for p in _TRIGGERS if (m := p.search(text))]
    head = text[: min(starts)] if starts else text
    head = _WHEN.sub("", head)
    head = _FILLER_HEAD.sub("", head)
    head = _LOCAL_WORDS.sub("", head)
    city = head.strip("的在去呢啊吧")
    when = m.group(0) if (m := _WHEN.search(text)) else ""
    rain = ""
    if _RAIN.search(text):
        rain = "雪" if "雪" in text else "雨"
    return {"city": city, "when": when, "rain": rain}


# ---------------------------------------------------------------------- 口语化


def spoken_temp(t: int) -> str:
    """温度读法：-3 → “零下3”。"""
    return f"零下{-t}" if t < 0 else str(t)


def spoken_range(low: int, high: int) -> str:
    """温度范围：“12到25度”“零下3到5度”；相同时只说一个。"""
    low, high = min(low, high), max(low, high)
    if low == high:
        return f"{spoken_temp(low)}度"
    return f"{spoken_temp(low)}到{spoken_temp(high)}度"


def day_label(day: date, today: date) -> str:
    """今天 / 明天 / 后天 / 大后天，再往后用“周六”这样的星期。"""
    return _RELATIVE_DAYS.get((day - today).days, WEEK_NAMES[day.weekday()])


def spoken_day(label: str, cast: DayForecast) -> str:
    """一天的预报，例如“明天白天小雨，晚上中雨，18到24度”。白天晚上一样时只说一次。"""
    if cast.day_weather == cast.night_weather:
        weather = cast.day_weather
    else:
        weather = f"白天{cast.day_weather}，晚上{cast.night_weather}"
    return f"{label}{weather}，{spoken_range(cast.night_temp, cast.day_temp)}"


def spoken_live(city: str, live: LiveWeather) -> str:
    """实况，例如“北京现在多云，23度”。"""
    return f"{city}现在{live.weather}，{spoken_temp(live.temperature)}度"


def spoken_rain(kind: str, labeled: list[tuple[str, DayForecast]]) -> str:
    """回答“会不会下雨 / 下雪”，例如“明天有雨，记得带伞。”“看起来不会下雨。”"""
    hits = [label for label, c in labeled if kind in c.day_weather or kind in c.night_weather]
    if not hits:
        return f"看起来不会下{kind}。"
    tip = "，记得带伞" if kind == "雨" else ""
    return f"{'、'.join(hits)}有{kind}{tip}。"


def _forecast_data(cast: DayForecast) -> dict[str, Any]:
    """提供给 AI 的结构化预报。"""
    return {
        "date": cast.date.isoformat(),
        "week": WEEK_NAMES[cast.date.weekday()],
        "day_weather": cast.day_weather,
        "night_weather": cast.night_weather,
        "day_temp": cast.day_temp,
        "night_temp": cast.night_temp,
        "day_wind": cast.day_wind,
        "day_power": cast.day_power,
    }


@register_skill
class WeatherSkill(Skill):
    name = "weather"
    description = "查询天气实况和天气预报"
    # 最坏情况：解析地名（拆分重试）+ 实况 + 预报，单次请求超时 8 秒，留足余量
    timeout = 30
    patterns = {"query": [p.pattern for p in _TRIGGERS]}

    async def setup(self, ctx: SkillContext) -> None:
        # Key 缺失时不抛错：抛错会让框架直接禁用技能，用户只会听到“我还不会这个”；
        # 这里保留技能，等真正被问到时再播报“还没有配置 Key”
        settings = ctx.settings
        self._client = AmapClient(
            str(settings.get("api_key") or ""), timeout_s=float(settings.get("timeout_s", 8))
        )
        self._default_city = str(settings.get("default_city") or "").strip()
        if not self._client.has_key:
            ctx.logger.warning("未配置高德 Key（skills.settings.weather.api_key），天气查询不可用")
        # 追问城市时记住原来的时间和问法，以及追问的截止时间
        self._pending: tuple[datetime, str, str] | None = None

    # ------------------------------------------------------------------ 匹配

    def match(self, text: str) -> Intent | None:
        intent = super().match(text)
        if intent is not None:
            self._pending = None
            intent.slots = parse_query(text)
            return intent
        # 刚追问过城市：这句话像个地名就接着查
        pending, self._pending = self._pending, None
        if pending is not None and datetime.now() < pending[0]:
            city = parse_query(text)["city"]
            if _CITY_ANSWER.fullmatch(city):
                slots = {"city": city, "when": pending[1], "rain": pending[2]}
                return Intent(skill=self.name, name="query", text=text, slots=slots)
        return None

    # ------------------------------------------------------------------ 处理

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        """查询天气并生成口语回复。可取消（请求在 await 处响应 CancelledError）。"""
        city = str(intent.slot("city", "")).strip()
        when_text = str(intent.slot("when", ""))
        rain = str(intent.slot("rain", ""))

        if not self._client.has_key:
            return SkillResult(reply="还没有配置高德地图的 Key，暂时查不了天气。")

        try:
            place = await self._locate(city)
        except AmbiguousPlace as exc:
            self._ask_city(intent, when_text, rain)
            names = "、".join(p.name for p in exc.candidates[:5])
            return SkillResult(
                reply=str(exc), follow_up=True, data={"error": str(exc), "candidates": names}
            )
        except WeatherError as exc:
            return SkillResult(reply=str(exc), data={"error": str(exc)})

        if place is None:
            self._ask_city(intent, when_text, rain)
            return SkillResult(reply="你想查哪个城市的天气？", follow_up=True)

        try:
            return await self._report(place, parse_when(when_text, date.today()), rain)
        except WeatherError as exc:
            return SkillResult(reply=str(exc), data={"error": str(exc)})

    def _ask_city(self, intent: Intent, when_text: str, rain: str) -> None:
        """记住这次的问法，追问窗口内用户只说城市名也能接着查。AI 调用时不追问。"""
        if intent.source == "rule":
            deadline = datetime.now() + timedelta(seconds=_AWAIT_CITY_SECONDS)
            self._pending = (deadline, when_text, rain)

    async def _locate(self, city: str) -> Place | None:
        """确定要查的地方：用户说的城市 → 默认城市 → IP 定位；都没有时返回 None。

        Raises:
            AmbiguousPlace: 地名对应多个行政区。
            WeatherError: 找不到地名或请求失败。
        """
        if city:
            try:
                return await self._client.resolve(city)
            except AmbiguousPlace:
                raise
            except WeatherError:
                # “那北京明天呢”里的“那”是口头语，但“那曲”的“那”是地名，
                # 所以不在解析时去掉，查不到时再去掉重试
                if city.startswith("那") and len(city) > 2:
                    return await self._client.resolve(city[1:])
                raise
        if self._default_city:
            return await self._client.resolve(self._default_city)
        return await self._client.locate_by_ip()

    async def _report(self, place: Place, when: When, rain: str) -> SkillResult:
        """按查询时间组织播报内容。

        Raises:
            WeatherError: 请求失败。
        """
        today = date.today()
        data: dict[str, Any] = {"adcode": place.adcode}

        if when.kind == "now":
            live = await self._client.live(place.adcode)
            city = live.city or place.name
            data |= {"city": city, "live": asdict(live)}
            reply = spoken_live(city, live) + "。"
            if rain:
                reply += f"现在有{rain}。" if rain in live.weather else f"现在没有下{rain}。"
            return SkillResult(reply=reply, data=data)

        city, casts = await self._client.forecast(place.adcode)
        city = city or place.name
        by_date = {c.date: c for c in casts}
        data["city"] = city

        if when.kind == "days":
            labeled = [(day_label(c.date, today), c) for c in casts if c.date >= today]
            if not labeled:
                return SkillResult(reply="暂时没有查到天气预报。", data=data)
            data["forecast"] = [_forecast_data(c) for _, c in labeled]
            body = "；".join(spoken_day(label, c) for label, c in labeled)
            prefix = spoken_rain(rain, labeled) if rain else ""
            return SkillResult(reply=f"{prefix}{city}未来{len(labeled)}天：{body}。", data=data)

        offsets = (0,) if when.kind == "today" else when.offsets
        labeled = []
        for offset in offsets:
            day = today + timedelta(days=offset)
            if day in by_date:
                labeled.append((day_label(day, today), by_date[day]))
        if not labeled:
            last = max(by_date) if by_date else today
            days = (last - today).days
            return SkillResult(
                reply=f"目前只能查到未来{days}天的天气预报。"
                if days > 0
                else "暂时没有查到天气预报。",
                data=data,
            )
        data["forecast"] = [_forecast_data(c) for _, c in labeled]

        parts: list[str] = []
        if when.kind == "today":
            live = await self._client.live(place.adcode)
            data["live"] = asdict(live)
            parts.append(spoken_live(city, live))
            parts.extend(spoken_day(label, c) for label, c in labeled)
        else:
            parts.append(city + spoken_day(*labeled[0]))
            parts.extend(spoken_day(label, c) for label, c in labeled[1:])
        prefix = spoken_rain(rain, labeled) if rain else ""
        return SkillResult(reply=prefix + "。".join(parts) + "。", data=data)

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="get_weather",
                description=(
                    "查询天气实况和未来几天（约 4 天）的天气预报，包括天气现象、气温、风力。"
                    "用户问天气、温度、会不会下雨、要不要带伞、穿什么时调用"
                ),
                intent="query",
                args_schema=WeatherArgs,
            )
        ]
