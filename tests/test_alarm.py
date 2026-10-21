"""闹钟技能测试：设置 / 查询 / 取消 / 追问 / 持久化 / 响铃与停止。"""

from __future__ import annotations

import asyncio

from pinecone.core.task_manager import TaskKind
from pinecone.nlu.router import IntentRouter
from pinecone.skills.builtin.alarm.skill import AlarmSkill, extract_label
from tests.conftest import SkillEnv


async def _setup(env: SkillEnv) -> tuple[IntentRouter, AlarmSkill]:
    env.config.skills.enabled = ["alarm", "task_control"]
    await env.manager.load()
    skill = env.manager.get("alarm")
    assert isinstance(skill, AlarmSkill)
    return IntentRouter(env.manager), skill


def test_extract_label() -> None:
    assert extract_label("明天7点提醒我开会") == "开会"
    assert extract_label("明天早上叫我7点起床") == "起床"
    assert extract_label("设个7点的闹钟") == ""


async def test_set_list_cancel(skill_env: SkillEnv) -> None:
    router, _ = await _setup(skill_env)
    reply = await router.handle("明天早上7点叫我起床")
    assert reply.text == "好的，明天早上7点提醒你起床"
    reply = await router.handle("每个工作日7点半的闹钟")
    assert reply.text == "好的，已设置工作日早上7点半的闹钟"
    assert "10分钟后" in (await router.handle("十分钟后提醒我关火")).text

    reply = await router.handle("我有哪些闹钟")
    assert reply.text.startswith("你有3个闹钟")

    assert (await router.handle("取消起床的闹钟")).text == "好的，已取消明天早上7点的闹钟"
    assert (await router.handle("取消8点的闹钟")).text == "没有找到8点的闹钟"
    # 两个闹钟时不指定目标 → 追问
    reply = await router.handle("取消闹钟")
    assert reply.follow_up and "要取消哪一个" in reply.text
    assert (await router.handle("取消所有闹钟")).text == "好的，已取消2个闹钟"
    assert (await router.handle("闹钟列表")).text == "你还没有设置闹钟"


async def test_ask_time_when_missing(skill_env: SkillEnv) -> None:
    """没说时间时追问，下一句直接说时间即可。"""
    router, _ = await _setup(skill_env)
    reply = await router.handle("帮我设个闹钟")
    assert reply.follow_up and "几点" in reply.text
    assert (await router.handle("明天下午三点")).text == "好的，已设置明天下午3点的闹钟"


async def test_persist_across_restart(skill_env: SkillEnv) -> None:
    router, _ = await _setup(skill_env)
    await router.handle("明天早上7点叫我起床")
    await skill_env.manager.unload()
    await skill_env.manager.load()
    router = IntentRouter(skill_env.manager)
    assert (await router.handle("我有哪些闹钟")).text == "你有1个闹钟：明天早上7点，起床"


async def test_ring_and_stop(skill_env: SkillEnv) -> None:
    """响铃是前台任务：会播报提醒内容并循环播放铃声，“停止”可以终止。"""
    router, skill = await _setup(skill_env)
    await router.handle("十分钟后提醒我关火")
    alarm = skill._store.all()[0]
    await skill._on_fire(alarm)
    await asyncio.sleep(0.1)

    ringing = skill_env.tasks.list(TaskKind.FOREGROUND)
    assert [t.name for t in ringing] == ["闹钟响铃"]
    assert set(skill_env.dialog.spoken) == {"关火的时间到了"}
    assert skill_env.output.played > 0

    await router.handle("关闭闹钟")
    await asyncio.sleep(0.05)
    assert skill_env.tasks.list(TaskKind.FOREGROUND) == []

    # 响过之后可以贪睡
    assert (await router.handle("再睡五分钟")).text == "好的，5分钟后再叫你"
