"""高德 Web 服务客户端：行政区查询、IP 定位、天气查询。

在架构中的位置：
    只被天气技能（``skill.py``）使用，不依赖音频、对话等任何底层模块。
    天气接口只接受城市编码 ``adcode``，所以用户说的地名要先经过
    `行政区域查询 <https://lbs.amap.com/api/webservice/guide/api/district>`_ 换成编码；
    没说地名也没配默认城市时，用 `IP 定位 <https://lbs.amap.com/api/webservice/guide/api/ipconfig>`_
    得到出口 IP 所在城市。

取消：
    每次请求都在调用方的协程里用 ``httpx.AsyncClient`` 异步完成，技能任务被取消时
    ``CancelledError`` 会在 ``await`` 处抛出，连接随 ``async with`` 一起关闭。

安全：
    Key 只出现在请求参数里。httpx 的异常信息会带上完整 URL（含 Key），
    所以这里记录日志时只写异常类型和状态码，不打印异常文本。
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

logger = logging.getLogger(__name__)

AMAP_BASE = "https://restapi.amap.com/v3"

# 常见错误码 → 口语提示，完整列表见高德“错误码说明”
_ERRORS = {
    "10001": "高德地图的 Key 不正确或已经过期，暂时查不了天气。",
    "10003": "今天的天气查询次数用完了，明天再试吧。",
    "10004": "查得太频繁了，稍等一会儿再试。",
    "10009": "高德地图的 Key 类型不对，需要 Web 服务类型的 Key。",
}

# 天气接口支持省、市、区县三级；街道等更细的级别不支持
_WEATHER_LEVELS = {"province", "city", "district"}
_ADCODE = re.compile(r"\d{6}")
# 行政区名称的后缀。用户一般只说“杭州”“西湖”，比较时去掉后缀
_SUFFIX = re.compile(r"(特别行政区|自治州|自治县|地区|省|市|区|县|盟|旗)$")


class WeatherError(Exception):
    """查询失败。消息是可以直接播报给用户的中文口语。"""


class AmbiguousPlace(WeatherError):
    """地名对应多个行政区（例如全国有好几个“朝阳区”），需要用户说得更具体。"""

    def __init__(self, keyword: str, candidates: list[Place]) -> None:
        super().__init__(f"有好几个叫{keyword}的地方，请加上省份或城市再说一遍。")
        self.keyword = keyword
        self.candidates = candidates


@dataclass(frozen=True)
class Place:
    """一个可查天气的行政区。"""

    name: str
    """行政区名称，例如“杭州市”。直接给 adcode 时与 adcode 相同。"""
    adcode: str
    """6 位行政区编码，天气接口的 ``city`` 参数。"""
    level: str = ""
    """级别：province / city / district；未知时为空。"""


@dataclass(frozen=True)
class LiveWeather:
    """实况天气（每小时更新多次）。"""

    city: str
    weather: str
    """天气现象，例如“多云”。"""
    temperature: int
    """实时气温（摄氏度）。"""
    humidity: str
    wind_direction: str
    wind_power: str
    """风力级别，例如“≤3”。"""
    report_time: str


@dataclass(frozen=True)
class DayForecast:
    """某一天的预报（每天 8、11、18 点左右更新）。"""

    date: date
    week: int
    """星期几，1~7 对应周一到周日。"""
    day_weather: str
    night_weather: str
    day_temp: int
    night_temp: int
    day_wind: str
    day_power: str


def strip_suffix(name: str) -> str:
    """去掉行政区后缀：“杭州市”→“杭州”，“西湖区”→“西湖”。名字太短时保持原样。"""
    stripped = _SUFFIX.sub("", name)
    return stripped if len(stripped) >= 2 else name


def _to_int(value: Any) -> int:
    """高德的温度是字符串（偶尔带小数），统一转成整数。"""
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return 0


def _text(value: Any) -> str:
    """高德在没有数据时会返回空列表 ``[]`` 而不是空字符串，这里统一成字符串。"""
    return value if isinstance(value, str) else ""


class AmapClient:
    """高德 Web 服务的异步客户端，带内存缓存。

    缓存策略：
        - 地名 → 行政区：进程内一直保留（行政区划几乎不变）；
        - 天气：``weather_ttl_s`` 秒（实况每小时才更新几次，反复问不必重复请求）；
        - IP 定位：1 小时。

    非线程安全：只应在事件循环线程中使用。所有方法都可取消。
    """

    def __init__(
        self,
        key: str,
        *,
        timeout_s: float = 8.0,
        weather_ttl_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            key: 高德 Web 服务 Key，为空时所有请求都抛出 :class:`WeatherError`。
            timeout_s: 单次 HTTP 请求超时（秒）。
            weather_ttl_s: 天气结果缓存时长（秒）。
            clock: 单调时钟，测试时可替换。
        """
        self._key = key.strip()
        self._timeout_s = timeout_s
        self._weather_ttl_s = weather_ttl_s
        self._clock = clock
        self._places: dict[str, Place] = {}
        self._weather: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self._ip_place: tuple[float, Place] | None = None

    @property
    def has_key(self) -> bool:
        """是否配置了 Key。"""
        return bool(self._key)

    # ------------------------------------------------------------------ 底层请求

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """请求一个高德接口并检查 ``status``。

        Raises:
            WeatherError: 未配置 Key、网络失败、返回非 JSON 或 ``status != 1``。
            asyncio.CancelledError: 任务被取消。
        """
        if not self._key:
            raise WeatherError("还没有配置高德地图的 Key，暂时查不了天气。")
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.get(f"{AMAP_BASE}/{path}", params={**params, "key": self._key})
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("高德接口 %s 返回 HTTP %s", path, exc.response.status_code)
            raise WeatherError("天气服务暂时不可用，请稍后再试。") from exc
        except httpx.HTTPError as exc:
            # 不打印 str(exc)：其中可能带有含 Key 的 URL
            logger.warning("高德接口 %s 请求失败：%s", path, type(exc).__name__)
            raise WeatherError("网络不太好，暂时查不到天气。") from exc
        except ValueError as exc:
            logger.warning("高德接口 %s 返回的不是 JSON", path)
            raise WeatherError("天气服务返回的数据有问题，请稍后再试。") from exc

        if not isinstance(data, dict) or str(data.get("status")) != "1":
            info = data if not isinstance(data, dict) else data.get("info")
            code = str(data.get("infocode", "")) if isinstance(data, dict) else ""
            logger.warning("高德接口 %s 返回错误：%s %s", path, code, info)
            raise WeatherError(_ERRORS.get(code, "天气服务暂时不可用，请稍后再试。"))
        return data

    # ------------------------------------------------------------------ 地名解析

    async def resolve(self, name: str) -> Place:
        """把用户说的地名解析成行政区。

        依次尝试：6 位 adcode 直接使用 → 整个名字查行政区 →
        拆成“上级 + 下级”（如“北京朝阳”）在上级的下属区划里找。

        Raises:
            AmbiguousPlace: 对应多个行政区。
            WeatherError: 找不到或请求失败。
        """
        name = name.strip()
        if _ADCODE.fullmatch(name):
            return Place(name=name, adcode=name)
        cached = self._places.get(name)
        if cached is not None:
            return cached

        places = await self._search(name)
        if not places:
            places = await self._search_compound(name)
        if not places:
            raise WeatherError(f"没找到{name}这个地方，换个说法试试？")
        place = self._pick(name, places)
        self._places[name] = place
        return place

    async def _search(self, keyword: str) -> list[Place]:
        """按名称查行政区（不含下级），只保留天气接口支持的级别，按 adcode 去重。"""
        data = await self._get("config/district", {"keywords": keyword, "subdistrict": 0})
        return _flatten(data.get("districts") or [])

    async def _search_compound(self, name: str) -> list[Place]:
        """处理“北京朝阳”“杭州西湖区”这类“上级 + 下级”的说法。

        从短到长尝试把名字拆成上级和下级：查上级时带两层下属区划，
        在其中找名称（去掉后缀后）等于下级的行政区。拆分点最多 len-3 个，
        地名一般只有 4~6 个字，请求数可控。
        """
        for i in range(2, len(name) - 1):
            parent, child = name[:i], name[i:]
            data = await self._get("config/district", {"keywords": parent, "subdistrict": 2})
            parents = data.get("districts") or []
            if not parents:
                continue
            children = [
                p
                for top in parents
                for p in _flatten(top.get("districts") or [])
                if p.name == child or strip_suffix(p.name) == strip_suffix(child)
            ]
            if children:
                return children
        return []

    @staticmethod
    def _pick(name: str, places: list[Place]) -> Place:
        """从多个候选中选出唯一一个：只有一个直接用；多个时选名称完全对上的。

        Raises:
            AmbiguousPlace: 仍然无法区分（例如“朝阳”对应朝阳市、北京朝阳区、长春朝阳区）。
        """
        if len(places) == 1:
            return places[0]
        exact = [p for p in places if p.name == name]
        if len(exact) == 1:
            return exact[0]
        raise AmbiguousPlace(strip_suffix(name), places)

    async def locate_by_ip(self) -> Place | None:
        """按出口 IP 定位当前城市；局域网 / 海外 IP 定位不到时返回 None。

        Raises:
            WeatherError: 请求失败（网络、Key 等问题）。
        """
        now = self._clock()
        if self._ip_place is not None and now < self._ip_place[0]:
            return self._ip_place[1]
        data = await self._get("ip", {})
        adcode, city = _text(data.get("adcode")), _text(data.get("city"))
        if not _ADCODE.fullmatch(adcode):
            logger.info("IP 定位失败，没有返回城市编码")
            return None
        place = Place(name=city or adcode, adcode=adcode, level="city")
        self._ip_place = (now + 3600, place)
        return place

    # ------------------------------------------------------------------ 天气

    async def _weather_data(self, adcode: str, extensions: str) -> dict[str, Any]:
        """请求天气接口（带缓存）。``extensions``：base 实况 / all 预报。"""
        key = (adcode, extensions)
        now = self._clock()
        cached = self._weather.get(key)
        if cached is not None and now < cached[0]:
            return cached[1]
        data = await self._get("weather/weatherInfo", {"city": adcode, "extensions": extensions})
        self._weather[key] = (now + self._weather_ttl_s, data)
        return data

    async def live(self, adcode: str) -> LiveWeather:
        """查询实况天气。

        Raises:
            WeatherError: 请求失败或该地区没有数据。
        """
        data = await self._weather_data(adcode, "base")
        lives = data.get("lives") or []
        if not lives:
            raise WeatherError("没查到这个地方的天气。")
        item = lives[0]
        return LiveWeather(
            city=_text(item.get("city")),
            weather=_text(item.get("weather")),
            temperature=_to_int(item.get("temperature")),
            humidity=_text(item.get("humidity")),
            wind_direction=_text(item.get("winddirection")),
            wind_power=_text(item.get("windpower")),
            report_time=_text(item.get("reporttime")),
        )

    async def forecast(self, adcode: str) -> tuple[str, list[DayForecast]]:
        """查询预报，返回（城市名，按日期排列的预报，通常从今天起 3~4 天）。

        Raises:
            WeatherError: 请求失败或该地区没有数据。
        """
        data = await self._weather_data(adcode, "all")
        forecasts = data.get("forecasts") or []
        if not forecasts or not forecasts[0].get("casts"):
            raise WeatherError("没查到这个地方的天气预报。")
        casts: list[DayForecast] = []
        for c in forecasts[0]["casts"]:
            try:
                day = date.fromisoformat(_text(c.get("date")))
            except ValueError:
                continue
            casts.append(
                DayForecast(
                    date=day,
                    week=_to_int(c.get("week")) or day.isoweekday(),
                    day_weather=_text(c.get("dayweather")),
                    night_weather=_text(c.get("nightweather")),
                    day_temp=_to_int(c.get("daytemp")),
                    night_temp=_to_int(c.get("nighttemp")),
                    day_wind=_text(c.get("daywind")),
                    day_power=_text(c.get("daypower")),
                )
            )
        return _text(forecasts[0].get("city")), casts


def _flatten(districts: list[dict[str, Any]]) -> list[Place]:
    """把行政区列表（可能带嵌套下级）展开成 Place 列表，过滤级别并按 adcode 去重。"""
    seen: dict[str, Place] = {}
    stack = list(districts)
    while stack:
        d = stack.pop(0)
        level, adcode = _text(d.get("level")), _text(d.get("adcode"))
        if level in _WEATHER_LEVELS and _ADCODE.fullmatch(adcode) and adcode not in seen:
            seen[adcode] = Place(name=_text(d.get("name")), adcode=adcode, level=level)
        stack.extend(d.get("districts") or [])
    return list(seen.values())
