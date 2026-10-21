"""时间技能测试。"""

from datetime import datetime

from pinecone.nlu.router import IntentRouter
from pinecone.skills.builtin.clock.skill import spoken_date, spoken_time
from tests.conftest import SkillEnv


def test_spoken_time_and_date() -> None:
    """口语化时间：时段、12 小时制、整点、补零。"""
    assert spoken_time(datetime(2026, 10, 21, 15, 20)) == "下午3点20分"
    assert spoken_time(datetime(2026, 10, 21, 8, 0)) == "早上8点整"
    assert spoken_time(datetime(2026, 10, 21, 12, 5)) == "中午12点05分"
    assert spoken_time(datetime(2026, 10, 21, 0, 30)) == "凌晨12点30分"
    assert spoken_date(datetime(2026, 10, 21)) == "10月21日，星期三"


async def test_clock_routing(skill_env: SkillEnv) -> None:
    """“几点了”“明天星期几”命中时间技能。"""
    skill_env.config.skills.enabled = ["clock"]
    await skill_env.manager.load()
    router = IntentRouter(skill_env.manager)
    assert (await router.handle("现在几点了？")).text.startswith("现在是")
    assert (await router.handle("明天星期几")).text.startswith("明天是")
