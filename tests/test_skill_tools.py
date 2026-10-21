"""技能转 AI 工具测试：工具收集、固定槽位、参数传递、与 Agent 联动。"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from pinecone.agent import VoiceAgent
from pinecone.agent.tools.skill_tools import skill_tools
from pinecone.core.config import AppConfig
from pinecone.skills.builtin.custom_task.skill import CustomTaskSkill
from pinecone.tasks import TaskSpec
from tests.agent_fakes import ScriptedChatModel
from tests.conftest import SkillEnv


async def _load(env: SkillEnv) -> None:
    env.config.skills.enabled = ["clock", "alarm", "task_control", "custom_task"]
    env.config.skills.settings["custom_task"] = {"dirs": []}
    await env.manager.load()
    custom = env.manager.get("custom_task")
    assert isinstance(custom, CustomTaskSkill)
    custom.reload(
        [
            TaskSpec.model_validate(
                {
                    "name": "开灯",
                    "keywords": ["开灯"],
                    "description": "打开客厅灯",
                    "actions": [{"type": "speak", "text": "灯已打开"}],
                }
            )
        ]
    )


async def test_collect_and_invoke(skill_env: SkillEnv) -> None:
    await _load(skill_env)
    tools = {t.name: t for t in skill_tools(skill_env.manager)}
    assert {
        "get_current_time",
        "set_alarm",
        "list_tasks",
        "cancel_task",
        "stop_speaking",
        "run_task_1",
    } <= set(tools)

    result = json.loads(await tools["get_current_time"].ainvoke({}))
    assert result["reply"].startswith("现在是")

    # 自定义任务：固定槽位 task=开灯，speak 动作已经播报，reply 为空
    result = json.loads(await tools["run_task_1"].ainvoke({}))
    assert result["reply"] == "" and skill_env.dialog.spoken == ["灯已打开"]

    result = json.loads(await tools["set_alarm"].ainvoke({"when": "明天早上7点", "label": "起床"}))
    assert result["reply"] == "好的，明天早上7点提醒你起床"
    assert result["data"]["label"] == "起床"


async def test_agent_calls_skill_tool(skill_env: SkillEnv) -> None:
    """AI 通过工具设置闹钟，工具结果回传给模型。"""
    await _load(skill_env)
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "set_alarm", "args": {"when": "明天早上8点", "label": "开会"}, "id": "c1"}
        ],
    )
    model = ScriptedChatModel.from_script(call, "好的，明天早上8点提醒你开会。")
    agent = VoiceAgent(AppConfig(), model, tools=skill_tools(skill_env.manager))

    reply = await agent.handle("明天早上提醒我8点开会")
    assert "".join([p async for p in reply.stream]) == "好的，明天早上8点提醒你开会。"
    assert "set_alarm" in model.bound_tools
    tool_msg = next(m for m in model.received[-1] if isinstance(m, ToolMessage))
    assert "明天早上8点提醒你开会" in tool_msg.content

    alarm = skill_env.manager.get("alarm")
    assert [a.label for a in alarm._store.all()] == ["开会"]
