"""天气技能测试（用 httpx.MockTransport 模拟高德接口，不访问外网）。"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import httpx
import pytest

from pinecone.nlu.matcher import normalize
from pinecone.nlu.router import IntentRouter
from pinecone.skills.builtin.weather.skill import When, parse_query, parse_when, spoken_range
from tests.conftest import SkillEnv

TODAY = date.today()

DISTRICTS: dict[str, list[dict[str, Any]]] = {
    "北京": [
        {
            "name": "北京市",
            "adcode": "110000",
            "level": "province",
            "districts": [
                {
                    "name": "北京城区",
                    "adcode": "110100",
                    "level": "city",
                    "districts": [{"name": "朝阳区", "adcode": "110105", "level": "district"}],
                }
            ],
        }
    ],
    "朝阳": [
        {"name": "朝阳市", "adcode": "211300", "level": "city"},
        {"name": "朝阳区", "adcode": "110105", "level": "district"},
        {"name": "朝阳区", "adcode": "220104", "level": "district"},
    ],
}


def _cast(offset: int, day: str, night: str, high: int, low: int) -> dict[str, str]:
    d = TODAY + timedelta(days=offset)
    return {
        "date": d.isoformat(),
        "week": str(d.isoweekday()),
        "dayweather": day,
        "nightweather": night,
        "daytemp": str(high),
        "nighttemp": str(low),
        "daywind": "北",
        "daypower": "1-3",
    }


class FakeAmap:
    """模拟高德接口，记录请求。"""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.ip_adcode: Any = "330100"
        self.weather_status = "1"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        params = request.url.params
        path = request.url.path
        if path == "/v3/config/district":
            districts = DISTRICTS.get(params["keywords"], [])
            if params["subdistrict"] == "0":
                districts = [{k: v for k, v in d.items() if k != "districts"} for d in districts]
            return httpx.Response(200, json={"status": "1", "districts": districts})
        if path == "/v3/ip":
            return httpx.Response(
                200, json={"status": "1", "city": "杭州市", "adcode": self.ip_adcode}
            )
        if path == "/v3/weather/weatherInfo":
            if self.weather_status != "1":
                return httpx.Response(200, json={"status": "0", "infocode": "10001"})
            city = {"110000": "北京市", "330100": "杭州市"}.get(params["city"], "某地")
            if params["extensions"] == "base":
                live = {"city": city, "weather": "多云", "temperature": "23", "humidity": "40"}
                return httpx.Response(200, json={"status": "1", "lives": [live]})
            casts = [
                _cast(0, "晴", "晴", 25, 12),
                _cast(1, "小雨", "中雨", 24, 18),
                _cast(2, "多云", "阴", 20, -3),
                _cast(3, "晴", "晴", 22, 10),
            ]
            return httpx.Response(
                200, json={"status": "1", "forecasts": [{"city": city, "casts": casts}]}
            )
        return httpx.Response(404)

    def count(self, path: str) -> int:
        return sum(1 for r in self.requests if r.url.path == path)


@pytest.fixture
def amap(monkeypatch: pytest.MonkeyPatch) -> FakeAmap:
    fake = FakeAmap()
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(fake), **kw)
    )
    return fake


async def _router(skill_env: SkillEnv, **settings: Any) -> IntentRouter:
    skill_env.config.skills.enabled = ["weather"]
    skill_env.config.skills.settings["weather"] = {"api_key": "test-key", **settings}
    await skill_env.manager.load()
    return IntentRouter(skill_env.manager)


def test_parse_query() -> None:
    """从说法中抽出城市、时间和是否问下雨；本地词和客套话不当作城市。"""
    q = lambda text: parse_query(normalize(text))  # noqa: E731
    assert q("北京明天天气怎么样？") == {"city": "北京", "when": "明天", "rain": ""}
    assert q("明天北京的天气") == {"city": "北京", "when": "明天", "rain": ""}
    assert q("帮我查一下上海的天气") == {"city": "上海", "when": "", "rain": ""}
    assert q("后天杭州会不会下雨")["city"] == "杭州"
    assert q("后天杭州会不会下雨")["rain"] == "雨"
    assert q("现在外面多少度") == {"city": "", "when": "现在", "rain": ""}
    assert q("那曲今天天气")["city"] == "那曲"


def test_parse_when() -> None:
    """时间说法：相对日期、星期、周末、未来几天；“大后天”不能被识别成“后天”。"""
    thursday = date(2026, 10, 22)
    assert parse_when("", thursday) == When("today")
    assert parse_when("现在", thursday) == When("now")
    assert parse_when("大后天", thursday) == When("dates", (3,))
    assert parse_when("周六", thursday) == When("dates", (2,))
    assert parse_when("下周一", thursday) == When("dates", (4,))
    assert parse_when("周末", thursday) == When("dates", (2, 3))
    assert parse_when("周末", date(2026, 10, 25)) == When("dates", (0, 6))
    assert parse_when("未来几天", thursday) == When("days")
    assert parse_when("星期天", thursday) == When("dates", (3,))


def test_spoken_range() -> None:
    """温度范围读法：负数读“零下”，高低温顺序不对时自动调整。"""
    assert spoken_range(12, 25) == "12到25度"
    assert spoken_range(5, -3) == "零下3到5度"
    assert spoken_range(7, 7) == "7度"


async def test_weather_routing(skill_env: SkillEnv, amap: FakeAmap) -> None:
    """今天 / 明天 / 未来几天 / 下雨 的口语回复；没说城市时按 IP 定位。"""
    router = await _router(skill_env)

    reply = (await router.handle("今天天气怎么样")).text
    assert reply == "杭州市现在多云，23度。今天晴，12到25度。"

    reply = (await router.handle("北京明天天气")).text
    assert reply == "北京市明天白天小雨，晚上中雨，18到24度。"

    reply = (await router.handle("后天会下雨吗")).text
    assert reply.startswith("看起来不会下雨。杭州市后天白天多云，晚上阴，零下3到20度")

    reply = (await router.handle("未来几天天气")).text
    assert reply.startswith("杭州市未来4天：今天晴，12到25度；明天白天小雨")


async def test_weather_cache(skill_env: SkillEnv, amap: FakeAmap) -> None:
    """地名解析和天气结果都有缓存，重复询问不重复请求。"""
    router = await _router(skill_env)
    await router.handle("北京明天天气")
    await router.handle("北京明天天气")
    assert amap.count("/v3/config/district") == 1
    assert amap.count("/v3/weather/weatherInfo") == 1


async def test_weather_ambiguous_and_follow_up(skill_env: SkillEnv, amap: FakeAmap) -> None:
    """重名地名提示补充上级；接着说“北京朝阳”能在北京的下属区划中找到朝阳区。"""
    router = await _router(skill_env)
    reply = await router.handle("朝阳明天天气")
    assert "有好几个叫朝阳的地方" in (reply.text or "")
    assert reply.follow_up

    reply = await router.handle("北京朝阳")
    assert (reply.text or "").startswith("某地明天")
    weather = [r for r in amap.requests if r.url.path == "/v3/weather/weatherInfo"]
    assert weather[-1].url.params["city"] == "110105"


async def test_weather_ask_city_when_ip_fails(skill_env: SkillEnv, amap: FakeAmap) -> None:
    """IP 定位不到时追问城市，追问窗口内只说城市名即可。"""
    amap.ip_adcode = []
    router = await _router(skill_env)
    reply = await router.handle("明天天气")
    assert reply.text == "你想查哪个城市的天气？" and reply.follow_up
    assert (await router.handle("北京")).text == "北京市明天白天小雨，晚上中雨，18到24度。"


async def test_weather_default_city(skill_env: SkillEnv, amap: FakeAmap) -> None:
    """配置了默认城市时不做 IP 定位。"""
    router = await _router(skill_env, default_city="110000")
    assert (await router.handle("明天天气")).text.startswith("北京市明天")
    assert amap.count("/v3/ip") == 0


async def test_weather_errors(skill_env: SkillEnv, amap: FakeAmap) -> None:
    """找不到城市、Key 错误时用中文提示，不抛异常。"""
    router = await _router(skill_env)
    assert (await router.handle("火星天气")).text == "没找到火星这个地方，换个说法试试？"
    assert (await router.handle("那北京明天天气呢")).text.startswith("北京市明天")
    amap.weather_status = "0"
    assert "Key 不正确" in ((await router.handle("北京天气")).text or "")


async def test_weather_without_key(skill_env: SkillEnv, amap: FakeAmap) -> None:
    """未配置 Key 时技能仍然加载，问天气时提示配置，且不发请求。"""
    skill_env.config.skills.enabled = ["weather"]
    await skill_env.manager.load()
    router = IntentRouter(skill_env.manager)
    assert "还没有配置高德地图的 Key" in ((await router.handle("今天天气")).text or "")
    assert amap.requests == []


async def test_weather_not_triggered(skill_env: SkillEnv, amap: FakeAmap) -> None:
    """“空调温度调到26度”这类话不应命中天气技能。"""
    await _router(skill_env)
    assert skill_env.manager.match(normalize("把空调温度调到26度")) is None
    assert skill_env.manager.match(normalize("烤箱多少度")) is None
