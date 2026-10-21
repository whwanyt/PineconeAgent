"""task_control 技能测试。"""

import asyncio

from pinecone.core.task_manager import TaskKind
from pinecone.nlu.router import IntentRouter
from tests.conftest import SkillEnv


async def make_router(env: SkillEnv) -> IntentRouter:
    env.config.skills.enabled = ["task_control"]
    await env.manager.load()
    return IntentRouter(env.manager)


async def test_stop_interrupts_foreground_only(skill_env: SkillEnv) -> None:
    """“别说了”停止前台任务，后台任务继续。"""
    router = await make_router(skill_env)
    ring = skill_env.tasks.spawn(asyncio.sleep(10), name="闹钟响铃")
    music = skill_env.tasks.spawn(asyncio.sleep(10), name="音乐播放", kind=TaskKind.BACKGROUND)

    reply = await router.handle("别说了！")
    assert reply.text is None and skill_env.dialog.interrupted == 1
    await ring.wait()
    assert ring.status == "cancelled" and not music.done
    await skill_env.tasks.shutdown()


async def test_cancel_by_name_and_fallthrough(skill_env: SkillEnv) -> None:
    """“取消备份”取消匹配的任务；没有匹配任务时不认领，交给其他技能 / 兜底。"""
    router = await make_router(skill_env)
    backup = skill_env.tasks.spawn(asyncio.sleep(10), name="备份照片", kind=TaskKind.BACKGROUND)

    assert (await router.handle("取消备份")).text == "好的，已取消备份照片"
    await backup.wait()
    assert (await router.handle("取消明天的闹钟")).text == "抱歉，我还不会这个。"


async def test_list_and_cancel_all_inside_dialog_task(skill_env: SkillEnv) -> None:
    """在「对话」任务中执行“停止所有任务”：取消其他任务，但不会取消对话任务自己。"""
    router = await make_router(skill_env)
    skill_env.tasks.spawn(asyncio.sleep(10), name="下载", kind=TaskKind.BACKGROUND)

    async def dialog_turn(text: str) -> str | None:
        return (await router.handle(text)).text

    listed = await skill_env.tasks.run(dialog_turn("现在在做什么"), name="对话")
    assert listed.result == "正在运行：下载"

    stopped = await skill_env.tasks.run(dialog_turn("停止所有任务"), name="对话")
    assert stopped.status == "done" and stopped.result == "好的，已停止所有任务"
    await asyncio.sleep(0.01)
    assert skill_env.tasks.list() == []
