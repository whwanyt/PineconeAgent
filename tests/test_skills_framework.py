"""技能框架测试：归一化、插件发现、路由、后台任务取消、超时。"""

import asyncio
import shutil
from pathlib import Path

import pytest

from pinecone.core.config import PROJECT_ROOT
from pinecone.dialog import Reply
from pinecone.nlu.matcher import normalize
from pinecone.nlu.router import IntentRouter
from pinecone.skills import Intent, Skill, SkillContext, SkillResult, register_skill
from tests.conftest import SkillEnv


def test_normalize() -> None:
    """去标点、去中文周围空格、全角转半角、英文小写。"""
    assert normalize("播放 周杰伦 的歌。") == "播放周杰伦的歌"
    assert normalize("现在几点了？") == "现在几点了"
    assert normalize("Play  Music！") == "play music"


def install_template(env: SkillEnv, tmp_path: Path) -> None:
    plugins = tmp_path / "plugins"
    shutil.copytree(PROJECT_ROOT / "pinecone" / "skills" / "_template", plugins / "hello")
    env.config.skills.plugin_dirs = [str(plugins)]
    env.config.skills.settings = {"hello": {"greeting": "插件你好"}}


async def test_plugin_discovery_and_routing(skill_env: SkillEnv, tmp_path: Path) -> None:
    """模板复制到插件目录后被自动加载；规则命中走技能，读取私有配置。"""
    install_template(skill_env, tmp_path)
    await skill_env.manager.load()
    assert skill_env.manager.get("hello") is not None

    router = IntentRouter(skill_env.manager)
    reply = await router.handle("你好！")
    assert reply.text == "插件你好" and reply.follow_up

    unknown = await router.handle("帮我订一张机票")
    assert unknown.text == "抱歉，我还不会这个。"


async def test_background_task_can_be_cancelled(skill_env: SkillEnv, tmp_path: Path) -> None:
    """技能启动的后台任务可以按名称取消。"""
    install_template(skill_env, tmp_path)
    await skill_env.manager.load()
    reply = await IntentRouter(skill_env.manager).handle("倒计时 100 秒")
    assert reply.text == "好的，开始100秒倒计时"

    [task] = skill_env.tasks.list()
    assert task.name == "倒计时"
    assert skill_env.tasks.cancel_by_name("倒计时")
    await task.wait()
    assert skill_env.dialog.spoken == []


@register_skill
class _SlowSkill(Skill):
    name = "test_slow"
    description = "慢技能"
    patterns = {"slow": [r"^慢慢来$"]}
    timeout = 0.05

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        await asyncio.sleep(10)
        return SkillResult(reply="不会到这里")


async def test_skill_timeout_and_fallback(skill_env: SkillEnv) -> None:
    """技能超时返回提示；未命中时交给兜底处理器。"""
    skill_env.config.skills.enabled = ["test_slow"]
    await skill_env.manager.load()

    class Fallback:
        async def handle(self, text: str, *, hint: str | None = None) -> Reply:
            return Reply(text=f"AI:{text}")

    router = IntentRouter(skill_env.manager, Fallback())
    assert (await router.handle("慢慢来")).text == "操作超时了，请稍后再试。"
    assert (await router.handle("讲个笑话")).text == "AI:讲个笑话"


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Tool:
    def __init__(self, result) -> None:
        self._result = result

    async def _run(self, **kwargs):
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result

    coroutine = _run


async def test_call_tool_extracts_text_and_reports_missing(skill_env: SkillEnv) -> None:
    """工具结果里的文本、内容块和附件二元组都能收成一段话；没有这个工具就报错。"""
    ctx = skill_env.make_context("demo")
    skill_env.tools["plain"] = _Tool("你好")
    skill_env.tools["blocks"] = _Tool([_Block("甲"), {"text": "乙"}])
    skill_env.tools["pair"] = _Tool(("正文", None))

    assert await ctx.call_tool("plain", {}) == "你好"
    assert await ctx.call_tool("blocks", {}) == "甲\n乙"
    assert await ctx.call_tool("pair", {}) == "正文"
    with pytest.raises(LookupError):
        await ctx.call_tool("missing", {})


async def test_call_tool_lets_cancellation_through(skill_env: SkillEnv) -> None:
    """打断时取消要传到外面，不能被收成普通错误文本。"""
    ctx = skill_env.make_context("demo")
    skill_env.tools["stop"] = _Tool(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await ctx.call_tool("stop", {})
