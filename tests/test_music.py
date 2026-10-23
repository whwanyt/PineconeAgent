"""音乐技能测试（用假播放器，不需要 mpv）。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from pinecone.core.events import StateChanged
from pinecone.core.state import AssistantState
from pinecone.dialog import Reply
from pinecone.nlu.router import IntentRouter
from pinecone.skills import Intent, SkillResult
from pinecone.skills.builtin.music import player as player_module
from pinecone.skills.builtin.music.library import MusicLibrary
from pinecone.skills.builtin.music.player import MusicPlayer, PlayState
from pinecone.skills.builtin.music.skill import MusicSkill, clean_query
from tests.agent_fakes import ScriptedChatModel
from tests.conftest import SkillEnv


class FakePlayer(MusicPlayer):
    """单曲假播放器。finish / fail 用来模拟一首结束或失败。"""

    def __init__(self) -> None:
        self.uri = ""
        self._state = PlayState.IDLE
        self._paused = False
        self.volume = 0
        self.closed = False
        self.loaded: list[str] = []

    def load(self, uri: str) -> None:
        self.uri = uri
        self.loaded.append(uri)
        self._state = PlayState.PLAYING
        self._paused = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self.uri = ""
        self._state = PlayState.IDLE
        self._paused = False

    def set_volume(self, volume: int) -> None:
        self.volume = volume

    @property
    def state(self) -> PlayState:
        return self._state

    @property
    def paused(self) -> bool:
        return self._paused

    def finish(self) -> None:
        """模拟这一首正常放完。"""
        if self._state is PlayState.PLAYING:
            self._state = PlayState.ENDED

    def fail(self) -> None:
        """模拟这一首播放失败。"""
        self._state = PlayState.FAILED

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
    monkeypatch.setattr("pinecone.skills.builtin.music.skill._POLL_SECONDS", 0.01)
    monkeypatch.setattr("pinecone.skills.builtin.music.skill._MIN_PLAY_SECONDS", 0)
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
    assert clean_query("音乐九万字") == "九万字"


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

    assert (await router.handle("播放周杰伦的歌")).text.startswith("好的，一共2首")
    assert len(skill._queue) == 2
    assert fake.uri == skill._queue.current.uri
    assert [t.name for t in skill_env.tasks.list()] == ["音乐播放"]

    await router.handle("下一首")
    assert skill._queue.position == 2
    assert (await router.handle("下一首")).text == "已经是最后一首了"
    assert (await router.handle("这是什么歌")).text == f"正在播放{skill._queue.current.label}"

    await router.handle("暂停")
    assert fake.paused
    await router.handle("继续播放")
    assert not fake.paused

    assert (await router.handle("大声点")).text == "音量75"
    assert (await router.handle("音量调到30")).text == "音量30"

    # 播放中说“停止”：停止音乐，后台任务结束
    await router.handle("停止")
    await asyncio.sleep(0.05)
    assert fake.state is PlayState.IDLE
    assert skill_env.tasks.list() == []


async def test_station_and_not_found(music) -> None:
    """没有 AI 时，本地找不到仍直接告诉用户。"""
    router, _, fake = music
    assert (await router.handle("播放中国之声")).text == "好的，播放中国之声"
    assert fake.uri == "http://example.com/zgzs.m3u8"
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


async def _agent_play_songs(skill: MusicSkill, songs: list[object]) -> SkillResult:
    """模拟 AI 把选中的歌曲对象交给 play_music。"""
    return await skill.handle(
        skill._ctx,
        Intent(skill="music", name="play", slots={"songs": songs}, source="agent"),
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
    assert fake.state is PlayState.IDLE


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
    assert fake.uri == str(new)
    assert any(track.uri == str(new) for track in skill._library.tracks)


async def test_agent_url_does_not_rescan(music, monkeypatch: pytest.MonkeyPatch) -> None:
    """音频 URL 直接播放，不重扫曲库。"""
    _, skill, fake = music
    calls = _count_rescan(monkeypatch, skill)
    url = "https://example.com/song.mp3"

    result = await _agent_play(skill, url)

    assert calls["n"] == 0
    assert result.reply == "好的，播放在线音乐"
    assert fake.uri == url


async def test_agent_name_miss_stays_with_tool(music, monkeypatch: pytest.MonkeyPatch) -> None:
    """AI 用歌名点播时不重扫；找不到只返回说明，不再交给 AI。"""
    _, skill, fake = music
    _ = skill._library.tracks
    calls = _count_rescan(monkeypatch, skill)

    result = await _agent_play(skill, "孙燕姿")

    assert calls["n"] == 0
    assert not result.delegate
    assert result.reply is not None and result.reply.startswith("本地曲库没有找到孙燕姿")
    assert fake.state is PlayState.IDLE


async def test_cancel_task_stops_music(music, skill_env: SkillEnv) -> None:
    """“停止所有任务”取消后台任务时，播放器一起停止。"""
    router, _, fake = music
    await router.handle("来首歌")
    assert fake.state is PlayState.PLAYING
    await router.handle("停止所有任务")
    await asyncio.sleep(0.05)
    assert fake.state is PlayState.IDLE


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


@pytest.mark.parametrize(
    ("text", "playing", "intent"),
    [
        ("下一首", False, None),
        ("下一首", True, "next"),
        ("下一首放九万字", True, "insert_next"),
        ("下一首是什么歌", True, "queue"),
        ("播放列表里有什么", False, "queue"),
        ("把晴天加到播放列表", False, "enqueue"),
        ("播放列表加上晴天", False, "enqueue"),
        ("清空播放列表", True, "clear_queue"),
        ("播放第三首", False, None),
        ("播放第三首", True, "jump"),
        ("删掉这首", True, "remove"),
        ("单曲循环", False, "mode_single"),
        ("循环播放这首歌", True, "mode_single"),
        ("循环播放", False, "mode_loop"),
        ("关闭单曲循环", False, "mode_order"),
        ("随机播放", False, "mode_shuffle"),
        ("随机播放周杰伦的歌", False, "play"),
        ("收藏这首歌", False, None),
        ("收藏这首歌", True, "favorite"),
        ("我喜欢周杰伦", False, None),
        ("删除跑步歌单", False, "delete_playlist"),
        ("有哪些歌单", False, "list_playlists"),
        ("播放我收藏的歌", False, "play_playlist"),
        ("播放周杰伦的歌单", False, "play_playlist"),
        ("播放周杰伦的歌", False, "play"),
        ("放大音量", True, "volume_up"),
        ("删除明天的闹钟", True, None),
    ],
)
async def test_phrase_routing(music, text: str, playing: bool, intent: str | None) -> None:
    """播放中才认领的口令，以及播放列表、歌单说法不要被点歌抢走。"""
    router, skill, _ = music
    if playing:
        await router.handle("来首歌")
    matched = skill.match(text)
    assert (None if matched is None else matched.name) == intent


async def test_queue_commands(music) -> None:
    """追加、插播、查看、跳转、删除和清空。"""
    router, skill, fake = music
    await router.handle("播放晴天")
    assert (await router.handle("把稻香加到播放列表")).text == "好的，已把稻香加到播放列表"
    assert [item.title for item in skill._queue.tracks] == ["晴天", "稻香"]
    assert (await router.handle("下一首放十年")).text.startswith("好的，下一首放")
    assert skill._queue.tracks[1].title == "陈奕迅 - 十年"
    assert "一共3首" in (await router.handle("播放列表里有什么")).text
    await router.handle("播放第三首")
    assert skill._queue.position == 3
    assert fake.uri == skill._queue.current.uri
    await router.handle("上一首")
    assert "已从播放列表删掉" in (await router.handle("删掉这首")).text
    assert skill._queue.current is not None
    assert (await router.handle("清空播放列表")).text == "好的，已清空播放列表，只留下正在放的这首"


async def test_modes_and_auto_advance(music, skill_env: SkillEnv) -> None:
    """列表循环会从最后一首回到第一首；顺序放完后任务结束。"""
    router, skill, fake = music
    await router.handle("播放晴天")
    await router.handle("把稻香加到播放列表")
    assert (await router.handle("列表循环")).text == "好的，列表循环"
    fake.finish()
    await asyncio.sleep(0.05)
    assert skill._queue.position == 2
    fake.finish()
    await asyncio.sleep(0.05)
    assert skill._queue.position == 1

    await router.handle("顺序播放")
    await router.handle("清空播放列表")
    fake.finish()
    await asyncio.sleep(0.05)
    assert skill_env.tasks.list() == []


async def test_three_failures_stop(music, skill_env: SkillEnv) -> None:
    """连续三首失败就停下并播报。"""
    router, _, fake = music
    await router.handle("来首歌")
    for _ in range(3):
        fake.fail()
        await asyncio.sleep(0.05)
    assert any("接连几首" in line for line in skill_env.dialog.spoken)
    assert skill_env.tasks.list() == []


async def test_favorites_survive_reload(music, skill_env: SkillEnv) -> None:
    """收藏写在数据库里，重新加载技能后还能播放。"""
    router, _, _ = music
    await router.handle("播放晴天")
    assert (await router.handle("收藏这首歌")).text == "好的，已收藏晴天"
    await skill_env.manager.unload()
    await skill_env.manager.load()
    skill = skill_env.manager.get("music")
    router = IntentRouter(skill_env.manager)
    reply = await router.handle("播放我收藏的歌")
    assert "我喜欢" in reply.text
    assert skill._queue.current is not None and skill._queue.current.title == "晴天"  # type: ignore[union-attr]


async def test_missing_playlist_delegates(music, skill_env: SkillEnv) -> None:
    """本地没有这张歌单时，把整单播放的办法告诉 AI。"""
    fallback = _Fallback()
    router = IntentRouter(skill_env.manager, fallback)
    reply = await router.handle("播放周杰伦的歌单")
    assert reply.text == "交给我了"
    assert fallback.hints[0] is not None
    assert "本地曲库没有找到周杰伦" in fallback.hints[0]
    assert "list_tool" in fallback.hints[0]


class _UrlTool:
    def __init__(
        self,
        urls: list[str] | None = None,
        *,
        missing: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.urls = urls or []
        self.calls = 0
        self.missing = missing
        self.error = error

    async def _run(self, **kwargs: object) -> str:
        self.calls += 1
        if self.missing:
            raise LookupError("get_audio_url")
        if self.error is not None:
            raise self.error
        song = kwargs["song"]
        assert isinstance(song, dict)
        url = (
            self.urls[self.calls - 1]
            if self.calls <= len(self.urls)
            else "https://cdn.example/a.mp3"
        )
        return '{"url":"' + url + '"}'

    coroutine = _run


def _song(name: str, song_id: str) -> dict[str, str]:
    return {"id": song_id, "source": "netease", "name": name, "artist": "黄诗扶"}


async def test_online_queue_resolves_and_prefetches(music, skill_env: SkillEnv) -> None:
    """播放多首在线歌时先取当前这首，并在播放中预取下一首。"""
    from pinecone.skills.builtin.music.online import UrlResolver

    _, skill, fake = music
    tool = _UrlTool(["https://cdn.example/1.mp3", "https://cdn.example/2.mp3"])
    skill_env.tools["get_audio_url"] = tool
    skill._url_tool = "get_audio_url"
    skill._resolver = UrlResolver(skill._ctx.call_tool, "get_audio_url", 1)
    result = await _agent_play_songs(skill, [_song("九万字", "1"), _song("稻香", "2")])
    assert result.reply is not None and "一共2首" in result.reply
    assert fake.uri == "https://cdn.example/1.mp3"
    await asyncio.sleep(0.05)
    assert tool.calls >= 2


async def test_online_failure_keeps_current_music(music, skill_env: SkillEnv) -> None:
    """新的在线歌取不到地址时，原来的歌继续放。"""
    router, skill, fake = music
    await router.handle("播放晴天")
    kept = fake.uri
    tool = _UrlTool(missing=True)
    skill_env.tools["get_audio_url"] = tool
    skill._url_tool = "get_audio_url"
    from pinecone.skills.builtin.music.online import UrlResolver

    skill._resolver = UrlResolver(skill._ctx.call_tool, "get_audio_url", 1)
    result = await _agent_play_songs(skill, [_song("九万字", "9")])
    assert result.reply == "音乐服务没有连上，在线歌曲暂时放不了"
    assert fake.uri == kept
    assert skill._queue.current.title == "晴天"


async def test_online_tool_error_stays_in_the_skill(music, skill_env: SkillEnv) -> None:
    """取地址工具报错时，技能自己收住，原来的歌继续放。"""
    router, skill, fake = music
    await router.handle("播放晴天")
    kept = fake.uri
    tool = _UrlTool(error=RuntimeError("no valid download url found or vip required"))
    skill_env.tools["get_audio_url"] = tool
    skill._url_tool = "get_audio_url"
    from pinecone.skills.builtin.music.online import UrlResolver

    skill._resolver = UrlResolver(skill._ctx.call_tool, "get_audio_url", 1)
    result = await _agent_play_songs(skill, [_song("九万字", "9")])
    assert result.reply == "音乐服务没有响应，在线歌曲暂时放不了"
    assert fake.uri == kept
    assert skill._queue.current.title == "晴天"


async def test_list_tool_outside_allowlist_is_rejected(music) -> None:
    """不在白名单里的取歌工具不能被模型拿来入队。"""
    _, skill, fake = music
    result = await skill.handle(
        skill._ctx,
        Intent(
            skill="music",
            name="play",
            slots={"list_tool": "music_get_playlist_songs", "list_args": {"id": "1"}},
            source="agent",
        ),
    )
    assert result.reply == "不能用这个方式取歌"
    assert fake.state is PlayState.IDLE


async def test_agent_song_dict_and_model(music, skill_env: SkillEnv) -> None:
    """songs 无论是字典还是模型，都由技能自己取地址。"""
    from pinecone.agent import VoiceAgent
    from pinecone.agent.tools.skill_tools import skill_tools
    from pinecone.skills.builtin.music.online import SongArg, UrlResolver

    _, skill, fake = music
    tool = _UrlTool(["https://cdn.example/chosen.mp3"])
    skill_env.tools["get_audio_url"] = tool
    skill._url_tool = "get_audio_url"
    skill._list_tools = {"music_search_songs"}
    skill._resolver = UrlResolver(skill._ctx.call_tool, "get_audio_url", 1)

    direct = await skill.handle(
        skill._ctx,
        Intent(
            skill="music",
            name="play",
            slots={"songs": [SongArg(id="9", source="netease", name="九万字", artist="黄诗扶")]},
            source="agent",
        ),
    )
    assert direct.reply == "好的，播放黄诗扶的九万字"
    assert fake.uri == "https://cdn.example/chosen.mp3"

    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "play_music",
                "args": {"songs": [_song("九万字", "9")]},
                "id": "c1",
            }
        ],
    )
    model = ScriptedChatModel.from_script(call, "好的，播放九万字。")
    agent = VoiceAgent(skill_env.config, model, tools=skill_tools(skill_env.manager))
    reply = await agent.handle("播放九万字")
    assert "九万字" in "".join([piece async for piece in reply.stream])
