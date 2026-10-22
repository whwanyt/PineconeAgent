"""音乐技能测试（用假播放器，不需要 mpv）。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from pinecone.core.events import StateChanged
from pinecone.core.state import AssistantState
from pinecone.dialog import Reply
from pinecone.nlu.router import IntentRouter
from pinecone.skills import Intent, SkillResult
from pinecone.skills.builtin.music import player as player_module
from pinecone.skills.builtin.music.library import MusicLibrary, Track
from pinecone.skills.builtin.music.player import MusicPlayer
from pinecone.skills.builtin.music.skill import MusicSkill, clean_query
from tests.conftest import SkillEnv


class FakePlayer(MusicPlayer):
    def __init__(self) -> None:
        self.tracks: list[Track] = []
        self.pos = 0
        self._paused = False
        self.volume = 0
        self.closed = False

    def play(self, tracks: list[Track]) -> None:
        self.tracks, self.pos, self._paused = list(tracks), 0, False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def next(self) -> bool:
        if self.pos + 1 >= len(self.tracks):
            return False
        self.pos += 1
        return True

    def previous(self) -> bool:
        if self.pos == 0:
            return False
        self.pos -= 1
        return True

    def stop(self) -> None:
        self.tracks = []

    def set_volume(self, volume: int) -> None:
        self.volume = volume

    @property
    def active(self) -> bool:
        return bool(self.tracks)

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def current(self) -> Track | None:
        return self.tracks[self.pos] if self.tracks else None

    def close(self) -> None:
        self.closed = True


def _make_library(root: Path) -> Path:
    for rel in ["周杰伦/晴天.mp3", "周杰伦/稻香.flac", "陈奕迅 - 十年.mp3", "readme.txt"]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


@pytest.fixture
async def music(skill_env: SkillEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakePlayer()
    monkeypatch.setattr(player_module, "create_player", lambda device=None: fake)
    library = _make_library(tmp_path / "music")
    skill_env.config.skills.enabled = ["music", "task_control"]
    skill_env.config.skills.settings["music"] = {
        "library": [str(library)],
        "stations": {"中国之声": "http://example.com/zgzs.m3u8"},
        "volume": 60,
    }
    await skill_env.manager.load()
    skill = skill_env.manager.get("music")
    assert isinstance(skill, MusicSkill)
    yield IntentRouter(skill_env.manager), skill, fake
    await skill_env.manager.unload()


def test_prepare_libmpv_search_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """macOS 上把 Homebrew 的 lib 目录加到 dyld 回退路径，且不丢掉原有目录。"""
    monkeypatch.setattr(player_module.sys, "platform", "darwin")
    monkeypatch.setattr(player_module.os.path, "isdir", lambda path: path == "/opt/homebrew/lib")
    # 空字符串会被当成“未设置”，同时让 monkeypatch 在测试结束后恢复原环境。
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "")

    player_module._prepare_libmpv_search_path()

    parts = os.environ["DYLD_FALLBACK_LIBRARY_PATH"].split(":")
    assert parts[0] == "/opt/homebrew/lib"
    assert "/usr/lib" in parts

    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/custom/lib")
    player_module._prepare_libmpv_search_path()
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"].split(":") == [
        "/opt/homebrew/lib",
        "/custom/lib",
    ]


def test_prepare_libmpv_search_path_skips_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux 由系统库路径解析 libmpv，不改环境变量。"""
    monkeypatch.setattr(player_module.sys, "platform", "linux")
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    player_module._prepare_libmpv_search_path()
    assert "DYLD_FALLBACK_LIBRARY_PATH" not in os.environ


def test_clean_query() -> None:
    assert clean_query("周杰伦的歌吧") == "周杰伦"
    assert clean_query("一首晴天") == "晴天"
    assert clean_query("音乐") == ""


def test_library_search(tmp_path: Path) -> None:
    library = MusicLibrary([_make_library(tmp_path)])
    assert len(library.tracks) == 3
    assert [t.title for t in library.search("晴天")] == ["晴天"]
    assert {t.title for t in library.search("周杰伦")} == {"晴天", "稻香"}
    assert [t.title for t in library.search("陈奕迅十年")] == ["陈奕迅 - 十年"]
    assert [t.title for t in library.search("周杰伦晴天")] == ["晴天"]
    assert library.search("不存在") == []


async def test_play_and_control(music, skill_env: SkillEnv) -> None:
    router, skill, fake = music
    # 没有音乐时“暂停”“下一首”不认领
    assert skill.match("暂停") is None

    assert (await router.handle("播放周杰伦的歌")).text.startswith("好的，播放")
    assert len(fake.tracks) == 2
    assert [t.name for t in skill_env.tasks.list()] == ["音乐播放"]

    await router.handle("下一首")
    assert fake.pos == 1
    assert (await router.handle("下一首")).text == "已经是最后一首了"
    assert (await router.handle("这是什么歌")).text == f"正在播放{fake.current.title}"

    await router.handle("暂停")
    assert fake.paused
    await router.handle("继续播放")
    assert not fake.paused

    assert (await router.handle("大声点")).text == "音量75"
    assert (await router.handle("音量调到30")).text == "音量30"

    # 播放中说“停止”：停止音乐，后台任务结束
    await router.handle("停止")
    await asyncio.sleep(0.05)
    assert not fake.active
    assert skill_env.tasks.list() == []


async def test_station_and_not_found(music) -> None:
    """没有 AI 时，本地找不到仍直接告诉用户。"""
    router, _, fake = music
    assert (await router.handle("播放中国之声")).text == "好的，播放中国之声"
    assert fake.current.uri == "http://example.com/zgzs.m3u8"
    assert (await router.handle("我想听孙燕姿的歌")).text == "曲库里没有找到孙燕姿"


class _Fallback:
    """记下交给 AI 的原话和 hint。"""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.hints: list[str | None] = []

    async def handle(self, text: str, *, hint: str | None = None) -> Reply:
        self.texts.append(text)
        self.hints.append(hint)
        return Reply(text="交给我了", follow_up=False)


def _count_rescan(monkeypatch: pytest.MonkeyPatch, skill: MusicSkill) -> dict[str, int]:
    """把重扫次数记下来，方便确认哪条路径扫了盘。"""
    calls = {"n": 0}
    original = skill._library.rescan

    def wrapped() -> int:
        calls["n"] += 1
        return original()

    monkeypatch.setattr(skill._library, "rescan", wrapped)
    return calls


async def _agent_play(skill: MusicSkill, query: str) -> SkillResult:
    """模拟 AI 调用 play_music，不经过路由器。"""
    return await skill.handle(
        skill._ctx,
        Intent(skill="music", name="play", slots={"query": query}, source="agent"),
    )


async def test_user_miss_delegates_without_rescan(
    music, skill_env: SkillEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """索引里没有的歌交给 AI，不重扫，也不开始播放。"""
    _, skill, fake = music
    _ = skill._library.tracks
    calls = _count_rescan(monkeypatch, skill)
    fallback = _Fallback()
    router = IntentRouter(skill_env.manager, fallback)

    reply = await router.handle("我想听孙燕姿的歌")

    assert reply.text == "交给我了"
    assert fallback.texts == ["我想听孙燕姿的歌"]
    assert len(fallback.hints) == 1
    assert fallback.hints[0] is not None
    assert "本地曲库没有找到孙燕姿" in fallback.hints[0]
    assert calls["n"] == 0
    assert not fake.active


async def test_agent_local_path_rescans_before_play(music, monkeypatch: pytest.MonkeyPatch) -> None:
    """AI 传入刚下载的本地路径时，先重扫再播放这个文件。"""
    _, skill, fake = music
    assert len(skill._library.tracks) == 3
    new = skill._library._folders[0] / "岁月如笔写春秋.mp3"
    new.write_bytes(b"")
    calls = _count_rescan(monkeypatch, skill)

    result = await _agent_play(skill, str(new))

    assert calls["n"] == 1
    assert not result.delegate
    assert result.reply == "好的，播放岁月如笔写春秋"
    assert fake.current is not None and fake.current.uri == str(new)
    assert any(track.uri == str(new) for track in skill._library.tracks)


async def test_agent_url_does_not_rescan(music, monkeypatch: pytest.MonkeyPatch) -> None:
    """音频 URL 直接播放，不重扫曲库。"""
    _, skill, fake = music
    calls = _count_rescan(monkeypatch, skill)
    url = "https://example.com/song.mp3"

    result = await _agent_play(skill, url)

    assert calls["n"] == 0
    assert result.reply == "好的，播放在线音乐"
    assert fake.current is not None and fake.current.uri == url


async def test_agent_name_miss_stays_with_tool(music, monkeypatch: pytest.MonkeyPatch) -> None:
    """AI 用歌名点播时不重扫；找不到只返回说明，不再交给 AI。"""
    _, skill, fake = music
    _ = skill._library.tracks
    calls = _count_rescan(monkeypatch, skill)

    result = await _agent_play(skill, "孙燕姿")

    assert calls["n"] == 0
    assert not result.delegate
    assert result.reply is not None and result.reply.startswith("本地曲库没有找到孙燕姿")
    assert not fake.active


async def test_cancel_task_stops_music(music, skill_env: SkillEnv) -> None:
    """“停止所有任务”取消后台任务时，播放器一起停止。"""
    router, _, fake = music
    await router.handle("来首歌")
    assert fake.active
    await router.handle("停止所有任务")
    await asyncio.sleep(0.05)
    assert not fake.active


async def test_ducking(music, skill_env: SkillEnv) -> None:
    """唤醒时压低音量，回到空闲后恢复。"""
    router, _, fake = music
    await router.handle("来首歌")
    assert fake.volume == 60
    await skill_env.bus.publish(
        StateChanged(old=AssistantState.IDLE, new=AssistantState.LISTENING, reason="wake")
    )
    await asyncio.sleep(0.01)
    assert fake.volume == 18
    await skill_env.bus.publish(
        StateChanged(old=AssistantState.SPEAKING, new=AssistantState.IDLE, reason="done")
    )
    await asyncio.sleep(0.01)
    assert fake.volume == 60
