"""custom_task 技能测试：路由、前台 / 后台执行、取消与 on_cancel、示例任务可加载。"""

from __future__ import annotations

import asyncio

from pinecone.core.config import PROJECT_ROOT
from pinecone.core.task_manager import TaskKind
from pinecone.nlu.router import IntentRouter
from pinecone.skills.builtin.custom_task.skill import CustomTaskSkill
from pinecone.tasks import TaskSpec, load_tasks
from tests.conftest import SkillEnv


def _spec(**data) -> TaskSpec:
    return TaskSpec.model_validate(data)


async def _setup(env: SkillEnv, *specs: TaskSpec) -> tuple[IntentRouter, CustomTaskSkill]:
    env.config.skills.enabled = ["custom_task", "task_control"]
    env.config.skills.settings["custom_task"] = {"dirs": []}
    await env.manager.load()
    skill = env.manager.get("custom_task")
    assert isinstance(skill, CustomTaskSkill)
    skill.reload(list(specs))
    return IntentRouter(env.manager), skill


def test_example_tasks_load() -> None:
    """仓库自带的示例任务都能通过校验。"""
    names = {t.name for t in load_tasks([PROJECT_ROOT / "tasks"])}
    assert {"开灯", "备份照片", "查询IP地址"} <= names


async def test_match_rules(skill_env: SkillEnv) -> None:
    _, skill = await _setup(
        skill_env,
        _spec(
            name="开灯",
            keywords=["开灯"],
            patterns=["打开(?P<room>客厅|卧室)的?灯"],
            actions=[{"type": "speak", "text": "x"}],
        ),
        _spec(name="开灯光秀", keywords=["开灯光秀"], actions=[{"type": "speak", "text": "x"}]),
        _spec(name="备份", keywords=["备份"], actions=[{"type": "speak", "text": "x"}]),
    )
    intent = skill.match("打开卧室灯")
    assert intent and intent.slots == {"room": "卧室", "task": "开灯"}
    # 多个关键词命中时取最长
    assert skill.match("开灯光秀").slots["task"] == "开灯光秀"
    # “取消备份”不会再触发备份
    assert skill.match("取消备份") is None
    assert skill.match("今天天气") is None


async def test_foreground_task(skill_env: SkillEnv) -> None:
    router, _ = await _setup(
        skill_env,
        _spec(
            name="开灯",
            patterns=["打开(?P<room>客厅|卧室)的?灯"],
            actions=[
                {"type": "shell", "command": "echo ok"},
                {"type": "speak", "text": "{{room}}灯已打开"},
            ],
            reply="结果{{output}}",
        ),
        _spec(name="坏任务", keywords=["坏任务"], actions=[{"type": "shell", "command": "exit 1"}]),
    )
    reply = await router.handle("打开客厅的灯")
    assert skill_env.dialog.spoken == ["客厅灯已打开"]
    assert reply.text == "结果ok"
    assert (await router.handle("坏任务")).text == "坏任务执行失败"


async def test_foreground_timeout_runs_on_cancel(skill_env: SkillEnv) -> None:
    router, _ = await _setup(
        skill_env,
        _spec(
            name="慢任务",
            keywords=["慢任务"],
            timeout=0.1,
            actions=[{"type": "sleep", "seconds": 5}],
            on_cancel=[{"type": "speak", "text": "已停止，{{reason}}"}],
        ),
    )
    reply = await router.handle("慢任务")
    assert reply.text is None
    assert skill_env.dialog.spoken == ["已停止，超时"]


async def test_background_task_and_cancel(skill_env: SkillEnv) -> None:
    router, _ = await _setup(
        skill_env,
        _spec(
            name="备份照片",
            keywords=["备份照片"],
            background=True,
            timeout=None,
            actions=[{"type": "shell", "command": "sleep 30"}],
            on_cancel=[{"type": "speak", "text": "已停止备份"}],
        ),
        _spec(
            name="快任务",
            keywords=["快任务"],
            background=True,
            done_reply="完成了{{output}}",
            actions=[{"type": "shell", "command": "echo 1"}],
        ),
    )
    assert (await router.handle("备份照片")).text == "好的，开始备份照片"
    tasks = skill_env.tasks.list(TaskKind.BACKGROUND)
    assert [(t.name, t.source) for t in tasks] == [("备份照片", "task:备份照片")]
    # 同名任务不重复启动
    assert (await router.handle("备份照片")).text == "备份照片正在进行中"

    # “停止”只停前台，后台任务继续
    await router.handle("停止")
    assert len(skill_env.tasks.list(TaskKind.BACKGROUND)) == 1

    assert (await router.handle("取消备份照片")).text == "好的，已取消备份照片"
    await asyncio.sleep(0.2)
    assert skill_env.tasks.list(TaskKind.BACKGROUND) == []
    assert skill_env.dialog.spoken == ["已停止备份"]

    await router.handle("快任务")
    await asyncio.sleep(0.3)
    assert skill_env.dialog.spoken[-1] == "完成了1"


async def test_tools(skill_env: SkillEnv) -> None:
    _, skill = await _setup(
        skill_env,
        _spec(
            name="开灯",
            keywords=["开灯"],
            description="打开灯",
            actions=[{"type": "speak", "text": "x"}],
        ),
        _spec(
            name="内部",
            keywords=["内部"],
            expose_to_agent=False,
            actions=[{"type": "speak", "text": "x"}],
        ),
    )
    tools = skill.tools()
    assert [(t.name, t.fixed_slots) for t in tools] == [("run_task_1", {"task": "开灯"})]
    assert "打开灯" in tools[0].description
