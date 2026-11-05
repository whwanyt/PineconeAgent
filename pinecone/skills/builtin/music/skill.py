"""音乐技能。

说法示例：
    播放   “播放周杰伦的歌”“我想听晴天”“来首歌”“随机播放”
    列表   “播放列表里有什么”“把晴天加到播放列表”“下一首放十年”“播放第三首”
    模式   “单曲循环”“列表循环”“顺序播放”“随机播放”
    歌单   “收藏这首歌”“播放我喜欢的歌”“新建跑步歌单”“有哪些歌单”
    控制   “暂停”“继续播放”“下一首”“上一首”“停止播放”“这是什么歌”
    音量   “大声点”“小声点”“音量调到50”“放大音量”

播放列表在 :class:`PlayQueue` 里。mpv 只播当前一首。在线歌只保存歌曲信息，
快轮到时用配置的 ``url_tool`` 取地址，并预取下一首。自建歌单存在
``data/music/playlists.db``，同样不存地址。

私有配置（config.yaml → skills.settings.music）::

    library: [~/Music]
    stations: {}
    volume: 60
    duck_ratio: 0.3
    audio_device: null
    mode: 顺序播放
    queue_limit: 100
    url_tool: ""
    list_tools: []
    online_timeout_s: 8
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pinecone.core.events import PlaylistSnapshot, PlaylistUpdated
from pinecone.core.task_manager import RunningTask
from pinecone.skills import Intent, Skill, SkillContext, SkillResult, ToolSpec, register_skill

from . import player as player_module
from .library import MusicLibrary, Track
from .online import (
    ResolveUnavailable,
    UrlResolver,
    preload_url,
    songs_from_result,
    track_from_song,
)
from .player import MusicPlayer, PlayState
from .playqueue import PlayMode, PlayQueue
from .store import FAVORITES, PlaylistStore, canonical_name
from .tools import music_tools

# 需要正在播放才认领。没在播时交给别的技能或 AI。
_WHILE_PLAYING = {
    "stop",
    "pause",
    "resume",
    "next",
    "previous",
    "what",
    "volume_up",
    "volume_down",
    "volume_set",
    "jump",
    "remove",
    "favorite",
    "unfavorite",
    "add_current",
    "remove_current_playlist",
}
_DUCK_STATES = {"listening", "thinking", "speaking", "follow_up"}
_VOLUME_STEP = 15
_POLL_SECONDS = 0.25
_MIN_PLAY_SECONDS = 1.0
_MAX_FAILURES = 3
_START_DEADLINE_S = 22.0
_ONLINE_LIMIT = 50
_SONG_ARG_LIMIT = 30
_CN_DIGIT = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


@dataclass
class Chosen:
    """一次点歌解析出的曲目，或一个已经可以返回的结果。"""

    tracks: list[Track] = field(default_factory=list)
    name: str = ""
    error: SkillResult | None = None


def clean_query(text: str) -> str:
    """从“周杰伦的歌吧”“一首晴天”“音乐九万字”中提取搜索词。"""
    query = re.sub(r"^(一首|首|一点|点|一些|些|一下|几首)", "", text)
    query = re.sub(r"^(音乐|歌曲)", "", query)
    query = re.sub(r"的?(歌曲|歌|音乐|专辑)?(吧|呗|啊|好吗)?$", "", query)
    return query.strip()


def is_stream_url(text: str) -> bool:
    """是否为可直接交给播放器的 http(s) 地址。"""
    return text.lower().startswith(("http://", "https://"))


def as_local_path(text: str) -> Path | None:
    """绝对路径或 ``~/`` 开头时返回展开后的路径，否则不是本地地址。"""
    if text.startswith("~") or Path(text).is_absolute():
        return Path(text).expanduser()
    return None


def parse_index(token: str) -> int | None:
    """把「3」或「三」变成从 1 开始的序号。"""
    if token.isdigit():
        value = int(token)
        return value or None
    return _CN_DIGIT.get(token)


@register_skill
class MusicSkill(Skill):
    """本地曲库、在线歌曲和自建歌单。"""

    name = "music"
    description = "播放本地音乐、在线音乐和网络电台，管理播放列表和歌单"
    priority = 95
    timeout = 25
    patterns = {
        "stop": [
            r"^(停止|关闭|关掉|退出|结束)(播放|音乐|歌曲|放歌)$",
            r"(别|不要)(放|播|唱)了",
            r"^(停止|停|停下|关掉)吧?$",
        ],
        "pause": [r"^暂停(播放|音乐|一下)?$"],
        "resume": [r"^(继续|恢复)(播放|放|音乐)?$"],
        "queue": [
            r"播放列表(里|中)?(有什么|有哪些|是什么|还有什么|还有哪些)",
            r"(还有|后面|接下来)(哪些|几首|什么)歌",
            r"下一首是什么歌",
            r"^播放列表$",
            r"^现在的播放列表$",
        ],
        "insert_next": [
            r"^(下一首|下一曲)(播放|放|播)(?P<query>.+)$",
            r"^插播(?P<query>.+)$",
        ],
        "enqueue": [
            r"^(把)?(?P<query>.+?)(加到|添加到|加入)播放列表$",
            r"^播放列表(加上|加入|添加)(?P<query>.+)$",
        ],
        "clear_queue": [r"^清空播放列表$"],
        "jump": [r"^(播放|跳到|切到)第(?P<index>\d+|[一二两三四五六七八九十])首$"],
        "remove": [
            r"^(删除|去掉|移除|删掉)(这首|这一首|当前这首)$",
            r"^(删除|去掉|移除|删掉)下一首$",
            r"^(删除|去掉|移除|删掉)第(?P<index>\d+|[一二两三四五六七八九十])首$",
            r"^(把)?(?P<query>.+?)从播放列表(里|中)?(删除|去掉|移除)$",
        ],
        "mode_order": [r"(取消|关闭|停止)(单曲)?循环", r"^顺序播放$"],
        "mode_unshuffle": [r"^(取消|关闭)随机(播放)?$"],
        "mode_single": [r"^单曲循环$", r"^循环播放这首歌$"],
        "mode_loop": [r"(列表|歌单|全部)循环", r"^循环播放$"],
        "mode_shuffle": [r"^(开启|打开|切换到)?随机播放(音乐|歌曲)?$"],
        "favorite": [r"^(收藏|喜欢)(这首歌|这一首|这首)$", r"^我喜欢这首歌$"],
        "unfavorite": [r"^取消收藏$", r"^不喜欢了$"],
        "save_playlist": [r"^把播放列表保存为(?P<name>.+)$"],
        "create_playlist": [r"^新建(一个)?(?P<name>.+?)歌单$", r"^新建歌单(?P<name>.+)$"],
        "remove_current_playlist": [r"^把这首从(?P<name>.+?)歌单(里|中)?(删除|去掉|移除)$"],
        "remove_playlist_track": [r"^从(?P<name>.+?)歌单(里|中)?(删除|去掉|移除)(?P<query>.+)$"],
        "add_current": [r"^把这首(歌)?加到(?P<name>.+?)歌单$", r"^加到(?P<name>.+?)歌单$"],
        "add_playlist": [r"^把(?P<query>.+?)加到(?P<name>.+?)歌单$"],
        "delete_playlist": [r"^(删除|删掉)(?P<name>.+?)歌单$", r"^把(?P<name>.+?)歌单删了$"],
        "list_playlists": [r"有哪些歌单", r"^我的歌单$", r"^歌单列表$"],
        "playlist_info": [
            r"^(?P<name>.+?)歌单(里|中)有(什么|哪些)歌$",
            r"^我收藏了哪些歌$",
            r"^我喜欢(里|中)?有(什么|哪些)歌$",
        ],
        "play_playlist": [
            r"^(?P<lead>随机)?播放(?P<name>.+?)的?歌单$",
            r"^(?P<lead>随机)?(播放|放)我(收藏|喜欢)的歌$",
        ],
        "next": [r"下一首|下一曲|切歌|换一首|换首歌|换一个"],
        "previous": [r"上一首|上一曲"],
        "what": [r"(这是|现在放的是|正在放的是|放的是)(什么|哪首|啥)歌?", r"这首?歌叫什么"],
        "volume_set": [r"音量(调到|设为|设置为|调成|调为)(?P<level>\d+)"],
        "volume_up": [
            r"(声音|音量)(大|高|调大|调高)一?点",
            r"大声一?点",
            r"放大(声音|音量)",
            r"调大(声音|音量)",
        ],
        "volume_down": [
            r"(声音|音量)(小|低|调小|调低)一?点",
            r"小声一?点",
            r"调小(声音|音量)",
        ],
        "play": [
            r"^(?P<lead>随机|单曲循环|列表循环|循环)?(播放|我想听|我要听|想听|来一?首|放一?首|放一?点|听一?点|来一?点|随便放|放一?些)(?P<query>.*)$",
            r"^(放|听)(歌|音乐)$",
        ],
    }

    async def setup(self, ctx: SkillContext) -> None:
        self._ctx = ctx
        settings = ctx.settings
        self._library = MusicLibrary(
            settings.get("library", ["~/Music"]), settings.get("stations", {})
        )
        self._volume = int(settings.get("volume", 60))
        self._duck_ratio = float(settings.get("duck_ratio", 0.3))
        self._ducked = False
        self._queue_limit = max(1, int(settings.get("queue_limit", 100)))
        try:
            mode = PlayMode(str(settings.get("mode", PlayMode.ORDER)))
        except ValueError:
            mode = PlayMode.ORDER
        self._queue = PlayQueue(mode, limit=self._queue_limit)
        self._url_tool = str(settings.get("url_tool", "") or "")
        self._list_tools = {str(name) for name in settings.get("list_tools", []) or []}
        timeout_s = float(settings.get("online_timeout_s", 8))
        self._resolver = UrlResolver(ctx.call_tool, self._url_tool, timeout_s)
        self._store = PlaylistStore(ctx.data_dir / "playlists.db")
        self._task: RunningTask | None = None
        self._lock = asyncio.Lock()
        self._session = 0
        self._failures = 0
        self._started_at = 0.0
        self._refetched: set[str] = set()
        self._pending_uri = ""
        self._player: MusicPlayer | None
        try:
            self._player = player_module.create_player(settings.get("audio_device"))
            self._apply_volume()
        except Exception as exc:  # 没装 libmpv 等
            ctx.logger.warning("音乐播放器不可用（%s），请安装 mpv / libmpv", exc)
            self._player = None
        self._unsubscribe = ctx.on_state_change(self._on_state_change)
        # 和当前快照相同的变化不再发布，避免 0.25 秒轮询把界面刷个不停
        self._playlist_snap = self.snapshot()

    async def teardown(self) -> None:
        self._unsubscribe()
        if self._player is not None:
            self._player.stop()
            self._player.close()
        self._store.close()

    @property
    def playing(self) -> bool:
        """是否有音乐任务在跑。切歌和取地址的间隙也算，避免控制口令被放走。"""
        return self._task is not None

    def match(self, text: str) -> Intent | None:
        intent = super().match(text)
        if intent is not None and intent.name in _WHILE_PLAYING and not self.playing:
            return None
        return intent

    def tools(self) -> list[ToolSpec]:
        return music_tools(url_tool=self._url_tool, list_tools=sorted(self._list_tools))

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        """处理音乐意图。队列或暂停状态变了就通知界面。

        可取消。取消过程中不再发布事件，避免把半截队列写出去。
        只在事件循环线程调用。
        """
        try:
            return await self._respond(ctx, intent)
        finally:
            task = asyncio.current_task()
            if task is None or not task.cancelling():
                await self._emit_playlist()

    def snapshot(self) -> PlaylistSnapshot:
        """当前播放队列。只读，不改播放位置。只在事件循环线程调用。"""
        return PlaylistSnapshot(
            tracks=tuple(item.label for item in self._queue.tracks),
            current=self._queue.position,
            mode=self._queue.mode.value,
            paused=bool(self._player and self._player.paused),
        )

    async def select(self, index: int) -> None:
        """从第 ``index`` 首开始放（从 1 数）。

        给界面点选用，不经过意图路由，因此不会打断对话，也不会记成一句用户话。
        没在播放时什么都不做。可取消。只在事件循环线程调用。
        """
        if self._player is None or self._queue.current is None:
            return
        await self._jump_to(index)
        await self._emit_playlist()

    async def _emit_playlist(self) -> None:
        """和上次发布的快照不同才发 :class:`PlaylistUpdated`。"""
        snap = self.snapshot()
        if snap == self._playlist_snap:
            return
        self._playlist_snap = snap
        await self._ctx.publish(
            PlaylistUpdated(
                tracks=snap.tracks,
                current=snap.current,
                mode=snap.mode,
                paused=snap.paused,
            )
        )

    async def _respond(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        if intent.name in {"play", "play_playlist", "enqueue", "insert_next"}:
            return await self._handle_play(intent)
        if intent.name in _WHILE_PLAYING and not self.playing:
            return SkillResult(reply="现在没有在放音乐")
        if self._player is None and intent.name not in _LIBRARY_INTENTS:
            return SkillResult(reply="音乐播放器不可用，请先安装 mpv")

        match intent.name:
            case "stop":
                self._stop()
                return SkillResult()
            case "pause":
                assert self._player is not None
                self._player.pause()
                return SkillResult()
            case "resume":
                assert self._player is not None
                self._player.resume()
                return SkillResult()
            case "next":
                return await self._skip(forward=True)
            case "previous":
                return await self._skip(forward=False)
            case "what":
                return self._what()
            case "queue":
                return SkillResult(reply=self._queue_reply(), data=self._status())
            case "clear_queue":
                return await self._clear_queue()
            case "jump":
                return await self._jump(intent)
            case "remove":
                return await self._remove(intent)
            case "edit":
                return await self._edit(intent)
            case (
                "mode"
                | "mode_single"
                | "mode_loop"
                | "mode_order"
                | "mode_shuffle"
                | "mode_unshuffle"
            ):
                return await self._apply_mode(self._mode_of(intent))
            case "volume_up" | "volume_down" | "volume_set":
                return self._change_volume(intent)
            case "favorite" | "add_current" | "add_playlist":
                return await self._add_playlist(intent)
            case "unfavorite":
                return self._remove_saved(intent)
            case "save_playlist":
                return self._save_playlist(str(intent.slot("name", "")))
            case "create_playlist":
                return self._create_playlist(str(intent.slot("name", "")))
            case "remove_current_playlist" | "remove_playlist_track":
                return self._remove_saved(intent)
            case "delete_playlist":
                return self._delete_playlist(str(intent.slot("name", "")))
            case "list_playlists" | "playlist_info":
                return self._describe_playlists(intent)
        return SkillResult()

    async def _handle_play(self, intent: Intent) -> SkillResult:
        """点歌、追加或插播。取列表不占播放锁，真正开播时才锁。"""
        if self._player is None:
            return SkillResult(reply="音乐播放器不可用，请先安装 mpv")
        self._apply_lead(intent)
        if intent.name == "play_playlist":
            chosen = self._from_playlist(self._playlist_name(intent), intent.source)
        else:
            chosen = await self._select(intent)
        if chosen.error is not None:
            return chosen.error
        nxt = intent.name == "insert_next" or bool(intent.slot("next", False))
        if intent.name == "play" or intent.name == "play_playlist" or not self.playing:
            return await self._activate(chosen.tracks, name=chosen.name)
        return await self._queue_add(chosen.tracks, nxt=nxt)

    def _apply_lead(self, intent: Intent) -> None:
        """「随机播放周杰伦」这类前缀先改模式，再换列表。"""
        lead = str(intent.slot("lead", "") or "")
        if lead == "随机":
            self._queue.set_mode(PlayMode.SHUFFLE)
        elif lead == "单曲循环":
            self._queue.set_mode(PlayMode.SINGLE)
        elif lead in {"循环", "列表循环"}:
            self._queue.set_mode(PlayMode.LOOP)

    async def _select(self, intent: Intent) -> Chosen:
        """按优先级取出要播的歌：整单工具、选中的一首、自建歌单、本地搜索。"""
        list_tool = str(intent.slot("list_tool", "") or "")
        if list_tool:
            return await self._from_list_tool(list_tool, intent.slots.get("list_args") or {})
        songs = intent.slots.get("songs") or []
        if songs:
            return self._from_songs(list(songs)[:_SONG_ARG_LIMIT])
        playlist = str(intent.slot("playlist", "") or "")
        if playlist:
            return self._from_playlist(playlist, intent.source)
        return self._from_query(str(intent.slot("query", "")), intent.source)

    async def _from_list_tool(self, tool: str, raw_args: object) -> Chosen:
        if tool not in self._list_tools:
            return Chosen(error=SkillResult(reply="不能用这个方式取歌"))
        args = dict(raw_args) if isinstance(raw_args, dict) else {}
        args.setdefault("limit", _ONLINE_LIMIT)
        try:
            text = await self._resolver.fetch_list(tool, args)
        except ResolveUnavailable as exc:
            return Chosen(error=SkillResult(reply=_offline_reply(exc)))
        tracks = _tracks_from_items(songs_from_result(text), self._resolver)
        if not tracks:
            return Chosen(error=SkillResult(reply="没有找到歌曲"))
        return Chosen(tracks=tracks[: self._queue_limit])

    def _from_songs(self, songs: list[object]) -> Chosen:
        tracks = _tracks_from_items(songs, self._resolver)
        if not tracks:
            return Chosen(error=SkillResult(reply="没有找到歌曲"))
        return Chosen(tracks=tracks)

    def _from_playlist(self, name: str, source: str) -> Chosen:
        matches = self._store.find_matches(name)
        if len(matches) > 1:
            names = "，".join(item.name for item in matches)
            return Chosen(error=SkillResult(reply=f"有{len(matches)}个歌单，{names}"))
        if len(matches) == 1:
            playlist = matches[0]
            tracks = self._store.tracks(playlist.id)
            if not tracks:
                return Chosen(error=SkillResult(reply=f"{playlist.name}歌单是空的"))
            return Chosen(tracks=tracks, name=playlist.name)
        label = canonical_name(name) or name
        return Chosen(error=self._missing(label, source, playlist=True))

    def _from_query(self, target: str, source: str) -> Chosen:
        """本地路径、地址、电台和歌名。只在事件循环线程调用。"""
        target = target.strip()
        if is_stream_url(target):
            return Chosen(tracks=[Track(title="在线音乐", uri=target)])
        local = as_local_path(target)
        if local is not None:
            self._library.rescan()
            if not local.is_file():
                return Chosen(error=self._missing(local.name, source))
            return Chosen(tracks=[Track(title=local.stem, uri=str(local))])
        query = clean_query(target)
        station = self._library.find_station(query) if query else None
        if station is not None:
            return Chosen(tracks=[station])
        tracks = self._library.search(query)
        if not tracks:
            return Chosen(error=self._missing(query, source))
        return Chosen(tracks=tracks)

    async def _activate(self, tracks: list[Track], *, name: str = "") -> SkillResult:
        """找到第一首能放的再替换队列。一首都没有时，原来的音乐继续。"""
        assert self._player is not None
        session = self._session
        deadline = time.monotonic() + _START_DEADLINE_S
        failed: list[Track] = []
        async with self._lock:
            ready, rest, failed, stopped = await self._first_playable(
                tracks, session, deadline, failed
            )
            if not self._alive(session):
                return SkillResult()
            if stopped is not None and ready is None:
                return stopped
            if ready is None:
                return SkillResult(reply=_none_reply(failed), data={"failed": _labels(failed)})
            admitted = self._queue.replace([ready, *rest])
            if not self._alive(session) or not await self._play_loaded(session):
                if self._alive(session):
                    self._queue.clear()
                return SkillResult(reply="这几首都拿不到播放地址", data={"failed": _labels(failed)})
            self._failures = 0
            self._ensure_watch()
            self._ctx.logger.info("播放：%s 等 %d 首", ready.label, len(admitted))
        reply = _started_reply(admitted, ready, failed, name)
        return SkillResult(reply=reply, data=self._status(failed))

    async def _first_playable(
        self,
        tracks: list[Track],
        session: int,
        deadline: float,
        failed: list[Track],
    ) -> tuple[Track | None, list[Track], list[Track], SkillResult | None]:
        """跳过开头放不了的歌。音乐服务不可用时停止这一批。"""
        for index, track in enumerate(tracks):
            if (
                not self._alive(session)
                or time.monotonic() > deadline
                or len(failed) >= _MAX_FAILURES
            ):
                break
            try:
                uri = await self._resolve(track)
            except ResolveUnavailable as exc:
                return None, [], failed, SkillResult(reply=_offline_reply(exc))
            if not self._alive(session):
                return None, [], failed, None
            if uri:
                self._pending_uri = uri
                return track, tracks[index + 1 :], failed, None
            failed.append(track)
        return None, [], failed, None

    async def _queue_add(self, tracks: list[Track], *, nxt: bool) -> SkillResult:
        async with self._lock:
            added = self._queue.insert_next(tracks) if nxt else self._queue.append(tracks)
        if not added:
            label = tracks[0].label if tracks else "这首歌"
            return SkillResult(reply=f"{label}已经在播放列表里了")
        if nxt:
            return SkillResult(reply=f"好的，下一首放{added[0].label}")
        if len(added) == 1:
            return SkillResult(reply=f"好的，已把{added[0].label}加到播放列表")
        return SkillResult(reply=f"好的，已加入{len(added)}首")

    async def _skip(self, *, forward: bool) -> SkillResult:
        self._failures = 0
        async with self._lock:
            moved = self._queue.next(auto=False) if forward else self._queue.previous()
            if moved is None:
                reply = "已经是最后一首了" if forward else "已经是第一首了"
                return SkillResult(reply=reply)
            self._pending_uri = ""
            if not await self._play_current(self._session):
                return SkillResult(reply="这首放不了")
        return SkillResult()

    async def _jump(self, intent: Intent) -> SkillResult:
        index = parse_index(str(intent.slot("index", "")))
        if index is None:
            return SkillResult(reply=f"播放列表只有{len(self._queue)}首")
        return await self._jump_to(index)

    async def _jump_to(self, position: int) -> SkillResult:
        async with self._lock:
            try:
                self._queue.jump(position)
            except IndexError as exc:
                return SkillResult(reply=str(exc))
            self._failures = 0
            self._pending_uri = ""
            if not await self._play_current(self._session):
                return SkillResult(reply="这首放不了")
        return SkillResult()

    async def _remove(self, intent: Intent) -> SkillResult:
        text = intent.text or ""
        if "下一首" in text:
            position = self._queue.position + 1
        elif intent.slot("index"):
            position = parse_index(str(intent.slot("index"))) or 0
        elif intent.slot("query"):
            found = self._queue.matches(str(intent.slot("query")))
            if len(found) > 1:
                spots = "、".join(f"第{item}首" for item in found)
                return SkillResult(reply=f"播放列表里有{len(found)}首，是{spots}")
            if not found:
                return SkillResult(reply="播放列表里没有这首")
            position = found[0]
        else:
            position = self._queue.position
        return await self._remove_at(position)

    async def _remove_at(self, position: int) -> SkillResult:
        async with self._lock:
            try:
                removed, was_current = self._queue.remove(position)
            except IndexError as exc:
                return SkillResult(reply=str(exc))
            if was_current:
                self._pending_uri = ""
                if self._queue.current is None or not await self._play_current(self._session):
                    self._stop_locked()
        return SkillResult(reply=f"好的，已从播放列表删掉{removed.label}")

    async def _edit(self, intent: Intent) -> SkillResult:
        action = str(intent.slot("action", ""))
        position = int(intent.slot("position", 0) or 0)
        if action == "clear":
            return await self._clear_queue()
        if position == 0:
            position = self._queue.position
        if action == "jump":
            return await self._jump_to(position)
        if action == "remove":
            return await self._remove_at(position)
        return SkillResult(reply="没有这个播放列表操作")

    async def _clear_queue(self) -> SkillResult:
        async with self._lock:
            if self._queue.current is None:
                return SkillResult(reply="播放列表是空的")
            self._queue.clear_others()
        return SkillResult(reply="好的，已清空播放列表，只留下正在放的这首")

    async def _apply_mode(self, mode: PlayMode) -> SkillResult:
        if mode is PlayMode.SHUFFLE and not self.playing:
            self._queue.set_mode(PlayMode.SHUFFLE)
            tracks = self._library.search("")
            if not tracks:
                return SkillResult(reply="曲库是空的，请把音乐放到音乐目录里")
            if self._player is None:
                return SkillResult(reply="音乐播放器不可用，请先安装 mpv")
            return await self._activate(tracks)
        async with self._lock:
            self._queue.set_mode(mode)
        return SkillResult(reply=f"好的，{mode.value}")

    def _mode_of(self, intent: Intent) -> PlayMode:
        if intent.name == "mode_single":
            return PlayMode.SINGLE
        if intent.name == "mode_loop":
            return PlayMode.LOOP
        if intent.name in {"mode_order", "mode_unshuffle"}:
            return PlayMode.ORDER
        if intent.name == "mode_shuffle":
            return PlayMode.SHUFFLE
        try:
            return PlayMode(str(intent.slot("mode", PlayMode.ORDER)))
        except ValueError:
            return PlayMode.ORDER

    def _what(self) -> SkillResult:
        current = self._queue.current
        reply = f"正在播放{current.label}" if current else "我也不知道这是什么歌"
        return SkillResult(reply=reply, data=self._status())

    def _queue_reply(self) -> str:
        current = self._queue.current
        if current is None:
            return "播放列表是空的"
        text = (
            f"播放列表一共{len(self._queue)}首，正在放第{self._queue.position}首，{current.label}"
        )
        upcoming = [item.label for item in self._queue.upcoming(5)]
        if upcoming:
            text += f"。接下来是{'、'.join(upcoming)}"
        if self._queue.mode is not PlayMode.ORDER:
            text += f"。{self._queue.mode.value}"
        return text

    async def _add_playlist(self, intent: Intent) -> SkillResult:
        name = str(intent.slot("name", "") or "")
        if intent.name == "favorite" or not name:
            name = FAVORITES
        if intent.slot("whole_queue"):
            tracks = self._queue.tracks
            if not tracks:
                return SkillResult(reply="播放列表是空的")
            return self._store_tracks(name, tracks, replace=False)
        if intent.name in {"favorite", "add_current"} or not _has_source(intent):
            tracks = _current_list(self._queue)
        else:
            chosen = await self._select(intent)
            if chosen.error is not None:
                return chosen.error
            tracks = chosen.tracks
        if not tracks:
            return SkillResult(reply="现在没有在放音乐")
        return self._store_tracks(name, tracks, replace=False)

    def _save_playlist(self, name: str) -> SkillResult:
        tracks = self._queue.tracks
        if not tracks:
            return SkillResult(reply="播放列表是空的")
        return self._store_tracks(name, tracks, replace=True)

    def _store_tracks(self, name: str, tracks: list[Track], *, replace: bool) -> SkillResult:
        matches = self._store.find_matches(name)
        if len(matches) > 1:
            names = "，".join(item.name for item in matches)
            return SkillResult(reply=f"有{len(matches)}个歌单，{names}")
        created = not matches
        playlist = matches[0] if matches else self._store.create(name)
        if replace:
            count = self._store.replace_tracks(playlist.id, tracks)
            return SkillResult(reply=f"好的，已把播放列表保存为{playlist.name}，一共{count}首")
        added = self._store.add(playlist.id, tracks)
        label = tracks[0].label
        if created and added:
            return SkillResult(reply=f"好的，新建了{playlist.name}歌单，加入了{label}")
        if added == 0:
            if playlist.name == FAVORITES:
                return SkillResult(reply=f"{label}已经收藏过了")
            return SkillResult(reply=f"{label}已经在{playlist.name}歌单里了")
        if playlist.name == FAVORITES and len(tracks) == 1:
            return SkillResult(reply=f"好的，已收藏{label}")
        return SkillResult(reply=f"好的，已把{label}加到{playlist.name}")

    def _create_playlist(self, name: str) -> SkillResult:
        if self._store.find(name) is not None:
            playlist = self._store.find(name)
            assert playlist is not None
            return SkillResult(reply=f"{playlist.name}歌单已经有了")
        if len(self._store.find_matches(name)) > 1:
            return SkillResult(reply="有多个同名的歌单")
        playlist = self._store.create(name)
        return SkillResult(reply=f"好的，新建了{playlist.name}歌单")

    def _remove_saved(self, intent: Intent) -> SkillResult:
        name = str(intent.slot("name", "") or FAVORITES)
        playlist = self._store.find(name)
        if playlist is None:
            return self._ambiguous_or_missing(name)
        position = int(intent.slot("position", 0) or 0)
        if position:
            removed = self._store.remove_at(playlist.id, position)
        elif intent.name == "remove_current_playlist" or intent.name == "unfavorite":
            current = self._queue.current
            if current is None:
                return SkillResult(reply="现在没有在放音乐")
            removed = current if self._store.remove(playlist.id, current.key) else None
        else:
            query = str(intent.slot("query", ""))
            tracks = self._store.tracks(playlist.id)
            found = [item for item in tracks if search_hit(query, item)]
            if len(found) != 1:
                return SkillResult(
                    reply="歌单里没有这首" if not found else "歌单里有多首，说一下第几首"
                )
            removed = found[0] if self._store.remove(playlist.id, found[0].key) else None
        if removed is None:
            return SkillResult(reply="歌单里没有这首")
        if playlist.name == FAVORITES:
            return SkillResult(reply=f"好的，已取消收藏{removed.label}")
        return SkillResult(reply=f"好的，已从{playlist.name}歌单删掉{removed.label}")

    def _delete_playlist(self, name: str) -> SkillResult:
        playlist = self._store.find(name)
        if playlist is None:
            return self._ambiguous_or_missing(name)
        try:
            self._store.delete(playlist.id)
        except ValueError as exc:
            return SkillResult(reply=str(exc))
        return SkillResult(reply=f"好的，已删除{playlist.name}歌单")

    def _describe_playlists(self, intent: Intent) -> SkillResult:
        text = intent.text or ""
        name = str(intent.slot("name", "") or "")
        if intent.name == "playlist_info" and not name:
            name = FAVORITES
        if "收藏" in text or "我喜欢" in text:
            name = name or FAVORITES
        if not name:
            playlists = self._store.all()
            body = "，".join(f"{item.name}{item.size}首" for item in playlists)
            return SkillResult(reply=f"你有{len(playlists)}个歌单：{body}")
        playlist = self._store.find(name)
        if playlist is None:
            return self._ambiguous_or_missing(name)
        tracks = self._store.tracks(playlist.id)[:20]
        if not tracks:
            return SkillResult(reply=f"{playlist.name}歌单是空的")
        names = "、".join(item.label for item in tracks)
        return SkillResult(reply=f"{playlist.name}里有{names}")

    def _ambiguous_or_missing(self, name: str) -> SkillResult:
        matches = self._store.find_matches(name)
        if len(matches) > 1:
            names = "，".join(item.name for item in matches)
            return SkillResult(reply=f"有{len(matches)}个歌单，{names}")
        return SkillResult(reply="没有这个歌单")

    def _playlist_name(self, intent: Intent) -> str:
        text = intent.text or ""
        if "收藏" in text or "喜欢" in text:
            return FAVORITES
        return str(intent.slot("name", "") or "")

    def _missing(self, query: str, source: str, *, playlist: bool = False) -> SkillResult:
        """本地没有。用户原话交给 AI；工具调用只返回说明。"""
        if source == "agent":
            if query:
                reply = (
                    f"本地曲库没有找到{query}。"
                    "一批歌或歌单可以传 list_tool，选中的一首传 songs，地址由这里自己取。"
                )
            else:
                reply = "曲库是空的，请把音乐放到音乐目录里。也可以传入本地文件路径或音频 URL。"
            return SkillResult(reply=reply)
        reply = f"曲库里没有找到{query}" if query else "曲库是空的，请把音乐放到音乐目录里"
        if playlist and query:
            hint = (
                f"本地曲库没有找到{query}。"
                "如果这是在线歌单、专辑、榜单或一批歌，用 play_music 的 list_tool 整单播放，不要逐首取地址"
            )
        elif query:
            hint = (
                f"本地曲库没有找到{query}，请用其他工具继续查找。"
                "选中的一首传 songs，一批歌或歌单传 list_tool，地址由音乐技能自己取"
            )
        else:
            hint = "曲库是空的，请用其他工具继续查找或下载，先用一句话告诉用户你在做什么"
        return SkillResult(reply=reply, delegate=True, hint=hint)

    def _stop(self) -> None:
        """停下播放。不拿锁：正在取地址的那次看到会话号变了就不会再加载。"""
        self._stop_locked()

    def _stop_locked(self) -> None:
        self._session += 1
        if self._player is not None:
            self._player.stop()
        self._queue.clear()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel("user")

    def _change_volume(self, intent: Intent) -> SkillResult:
        if intent.name == "volume_set":
            self._volume = int(intent.slot("level", self._volume))
        elif intent.name == "volume_up":
            self._volume += _VOLUME_STEP
        else:
            self._volume -= _VOLUME_STEP
        self._volume = max(5, min(100, self._volume))
        self._apply_volume()
        return SkillResult(reply=f"音量{self._volume}")

    async def _watch(self) -> None:
        """后台任务「音乐播放」：一首结束后按模式换下一首。取消时清掉队列。"""
        me = self._ctx.tasks.current()
        try:
            while True:
                await asyncio.sleep(_POLL_SECONDS)
                player = self._player
                if player is None:
                    return
                if player.state is PlayState.PLAYING:
                    await self._prefetch()
                    continue
                advanced = False
                async with self._lock:
                    state = player.state
                    if state is PlayState.PLAYING:
                        continue
                    if state is PlayState.IDLE or not await self._advance(state):
                        return
                    advanced = True
                # 锁外发布：订阅者若再调本技能，不会和播放锁死锁
                if advanced:
                    await self._emit_playlist()
        finally:
            stopped = self._task is me
            if stopped:
                self._task = None
                self._queue.clear()
                if self._player is not None:
                    self._player.stop()
            task = asyncio.current_task()
            if stopped and (task is None or not task.cancelling()):
                await self._emit_playlist()

    async def _prefetch(self) -> None:
        """预取下一首地址。不拿播放锁，失败就等真正轮到时再试。"""
        nxt = self._queue.peek_next()
        if nxt is None or not nxt.online or self._resolver.cached(nxt):
            return
        try:
            await self._resolver.resolve(nxt)
        except asyncio.CancelledError:
            raise
        except ResolveUnavailable:
            return

    async def _advance(self, state: PlayState) -> bool:
        """根据结束原因换歌。调用时已经拿着播放锁。"""
        track = self._queue.current
        if track is None:
            return False
        session = self._session
        if state is PlayState.FAILED or self._played_for() < _MIN_PLAY_SECONDS:
            self._failures += 1
            if await self._retry_online(track, session):
                return True
            self._drop_current()
        else:
            self._failures = 0
            if self._queue.next(auto=True) is None:
                return False
        if self._failures >= _MAX_FAILURES:
            self._ctx.announce("接连几首都放不了，先停下了")
            return False
        return await self._play_current(session)

    async def _retry_online(self, track: Track, session: int) -> bool:
        if not track.online or track.key in self._refetched:
            return False
        self._refetched.add(track.key)
        try:
            uri = await self._resolver.resolve(track, fresh=True)
        except ResolveUnavailable:
            return False
        if not uri or not self._alive(session):
            return False
        self._load(uri)
        return True

    async def _play_current(self, session: int) -> bool:
        """从当前曲开始，放不了的移出队列。调用时已经拿着播放锁。"""
        while self._queue.current is not None:
            if not self._alive(session):
                return False
            track = self._queue.current
            try:
                uri = self._pending_uri or await self._resolve(track)
            except ResolveUnavailable:
                return False
            self._pending_uri = ""
            if not self._alive(session):
                return False
            if uri:
                self._load(uri)
                return self._player is not None and self._player.state is not PlayState.FAILED
            self._failures += 1
            if self._failures >= _MAX_FAILURES:
                self._ctx.announce("接连几首都放不了，先停下了")
                return False
            self._drop_current()
        return False

    async def _play_loaded(self, session: int) -> bool:
        return await self._play_current(session)

    async def _resolve(self, track: Track, *, fresh: bool = False) -> str:
        if track.uri and not fresh:
            return track.uri
        return await self._resolver.resolve(track, fresh=fresh)

    async def _resolve_or_empty(self, track: Track) -> str:
        try:
            return await self._resolve(track)
        except ResolveUnavailable:
            return ""

    def _drop_current(self) -> None:
        position = self._queue.position
        if position:
            self._queue.remove(position)

    def _load(self, uri: str) -> None:
        assert self._player is not None
        self._player.load(uri)
        self._started_at = time.monotonic()

    def _played_for(self) -> float:
        if self._started_at <= 0:
            return 0
        return time.monotonic() - self._started_at

    def _ensure_watch(self) -> None:
        if self._task is None:
            self._task = self._ctx.spawn(self._watch(), name="音乐播放")

    def _alive(self, session: int) -> bool:
        return session == self._session

    def _apply_volume(self) -> None:
        if self._player is not None:
            ratio = self._duck_ratio if self._ducked else 1.0
            self._player.set_volume(round(self._volume * ratio))

    def _on_state_change(self, old: str, new: str) -> None:
        ducked = new in _DUCK_STATES
        if ducked != self._ducked:
            self._ducked = ducked
            if self.playing:
                self._apply_volume()

    def _status(self, failed: list[Track] | None = None) -> dict[str, object]:
        current = self._queue.current
        data: dict[str, object] = {
            "current": self._queue.position,
            "total": len(self._queue),
            "mode": self._queue.mode.value,
            "paused": bool(self._player and self._player.paused),
            "title": current.label if current else "",
            "upcoming": [item.label for item in self._queue.upcoming(10)],
            "tracks": [
                {"index": index, "title": item.label, "current": index == self._queue.position}
                for index, item in enumerate(self._queue.tracks, 1)
            ],
        }
        if failed:
            data["failed"] = _labels(failed)
        return data


_LIBRARY_INTENTS = {
    "create_playlist",
    "delete_playlist",
    "list_playlists",
    "playlist_info",
    "save_playlist",
    "add_playlist",
    "add_current",
    "favorite",
    "unfavorite",
    "remove_playlist_track",
    "remove_current_playlist",
}


def _tracks_from_items(items: list[object], resolver: UrlResolver) -> list[Track]:
    tracks: list[Track] = []
    for item in items:
        if not isinstance(item, dict) and not hasattr(item, "model_dump"):
            continue
        track = track_from_song(item)  # type: ignore[arg-type]
        if track is None:
            continue
        url = preload_url(item)  # type: ignore[arg-type]
        if url:
            resolver.remember(track, url)
        tracks.append(track)
    return tracks


def _has_source(intent: Intent) -> bool:
    """工具有没有另外指定要加入的歌。没有就用当前这首。"""
    if intent.slot("query") or intent.slot("list_tool") or intent.slot("playlist"):
        return True
    songs = intent.slots.get("songs") or []
    return bool(songs)


def _current_list(queue: PlayQueue) -> list[Track]:
    current = queue.current
    return [current] if current is not None else []


def _labels(tracks: list[Track]) -> list[str]:
    return [track.label for track in tracks]


def _started_reply(tracks: list[Track], first: Track, failed: list[Track], name: str) -> str:
    skipped = f"，有{len(failed)}首放不了已跳过" if failed else ""
    if name:
        return f"好的，播放{name}，一共{len(tracks)}首{skipped}"
    if len(tracks) > 1:
        return f"好的，一共{len(tracks)}首，先放{first.label}{skipped}"
    return f"好的，播放{first.label}{skipped}"


def _none_reply(failed: list[Track]) -> str:
    if failed:
        return "这几首都拿不到播放地址"
    return "没有找到歌曲"


def _offline_reply(exc: ResolveUnavailable) -> str:
    text = str(exc)
    if "没有连上" in text or "没有配置" in text:
        return "音乐服务没有连上，在线歌曲暂时放不了"
    return "音乐服务没有响应，在线歌曲暂时放不了"


def search_hit(query: str, track: Track) -> bool:
    """歌单里的歌名或歌手是否对得上这句话。"""
    if not query:
        return False
    return query in track.title or query in track.label or query in track.artist
