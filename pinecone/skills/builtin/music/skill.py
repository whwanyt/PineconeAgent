"""音乐技能。

说法示例：
    播放   “播放周杰伦的歌”“我想听晴天”“来首歌”“放点音乐”“播放中国之声”（电台）
    控制   “暂停”“继续播放”“下一首”“上一首”“停止播放”“这是什么歌”
    音量   “大声点”“小声点”“音量调到50”

设计要点：
    - 播放期间有一个 **后台** 托管任务「音乐播放」：唤醒、说话不会打断音乐；
      “取消音乐”“停止所有任务”或“停止播放”都会终止它并停止播放器；
    - 音乐播放时，“停止 / 暂停 / 下一首 / 大声点”等短指令由本技能认领（优先级高于 task_control）；
      没有音乐在播时这些指令不认领，交给其他技能；
    - 唤醒 / 思考 / 播报时自动把音乐压低（ducking），对话结束后恢复；
    - 用户点的歌本地没有时，不自己联网，把原话和 hint（本地没找到）交给 AI，
      由模型先开口说明正在做什么，再决定要不要用其他工具。AI 再调用播放时：
      本地文件路径会先重扫曲库再播，音频 URL 直接播、不重扫。

私有配置（config.yaml → skills.settings.music）::

    library: [~/Music]     # 音乐目录列表
    stations: {}           # 电台：名称 → 流地址
    volume: 60             # 默认音量（0~100）
    duck_ratio: 0.3        # 对话时音乐音量的比例
    audio_device: null     # mpv 输出设备（如 "alsa/plughw:1,0"），null 为系统默认
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from pydantic import BaseModel, Field

from pinecone.core.task_manager import RunningTask
from pinecone.skills import Intent, Skill, SkillContext, SkillResult, ToolSpec, register_skill

from . import player as player_module
from .library import MusicLibrary, Track
from .player import MusicPlayer

# 需要“正在播放音乐”才认领的意图
_CONTROL_INTENTS = {
    "stop",
    "pause",
    "resume",
    "next",
    "previous",
    "what",
    "volume_up",
    "volume_down",
    "volume_set",
}
# 这些状态下压低音乐（用户在说话 / 助手在思考或播报）
_DUCK_STATES = {"listening", "thinking", "speaking", "follow_up"}
_VOLUME_STEP = 15
# 连续多少次轮询发现播放器空闲才认为播完（避免切歌加载间隙被误判）
_IDLE_POLLS = 3
_POLL_SECONDS = 0.5


class PlayMusicArgs(BaseModel):
    query: str = Field(
        default="",
        description="歌名、歌手、电台名、本地文件路径或 http(s) 音频地址；歌名为空表示随机播放本地曲库",
    )


def clean_query(text: str) -> str:
    """从“周杰伦的歌吧”“一首晴天”中提取搜索词。"""
    query = re.sub(r"^(一首|首|一点|点|一些|些|一下|几首)", "", text)
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


@register_skill
class MusicSkill(Skill):
    name = "music"
    description = "播放本地音乐和网络电台"
    priority = 95
    timeout = 15
    patterns = {
        "stop": [
            r"^(停止|关闭|关掉|退出|结束)(播放|音乐|歌曲|放歌)$",
            r"(别|不要)(放|播|唱)了",
            r"^(停止|停|停下|关掉)吧?$",
        ],
        "pause": [r"^暂停(播放|音乐|一下)?$"],
        "resume": [r"^(继续|恢复)(播放|放|音乐)?$"],
        "next": [r"下一首|下一曲|切歌|换一首|换首歌|换一个"],
        "previous": [r"上一首|上一曲"],
        "what": [r"(这是|现在放的是|正在放的是|放的是)(什么|哪首|啥)歌?", r"这首?歌叫什么"],
        "volume_set": [r"音量(调到|设为|设置为|调成|调为)(?P<level>\d+)"],
        "volume_up": [r"(声音|音量)(大|高|调大|调高)一?点", r"大声一?点"],
        "volume_down": [r"(声音|音量)(小|低|调小|调低)一?点", r"小声一?点"],
        "play": [
            r"^(播放|我想听|我要听|想听|来一?首|放一?首|放一?点|听一?点|来一?点|随便放|放一?些)(?P<query>.*)$",
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
        self._task: RunningTask | None = None
        self._player: MusicPlayer | None
        try:
            self._player = player_module.create_player(settings.get("audio_device"))
            self._apply_volume()
        except Exception as exc:  # 没装 libmpv 等
            ctx.logger.warning("音乐播放器不可用（%s），请安装 mpv / libmpv", exc)
            self._player = None
        self._unsubscribe = ctx.on_state_change(self._on_state_change)

    async def teardown(self) -> None:
        self._unsubscribe()
        if self._player is not None:
            self._player.stop()
            self._player.close()

    # ------------------------------------------------------------------ 匹配

    @property
    def playing(self) -> bool:
        """是否有音乐在播（含暂停）。"""
        return self._player is not None and self._player.active

    def match(self, text: str) -> Intent | None:
        intent = super().match(text)
        if intent is not None and intent.name in _CONTROL_INTENTS and not self.playing:
            return None
        return intent

    # ------------------------------------------------------------------ 处理

    async def handle(self, ctx: SkillContext, intent: Intent) -> SkillResult:
        player = self._player
        if player is None:
            return SkillResult(reply="音乐播放器不可用，请先安装 mpv")
        if intent.name == "play":
            return self._play(intent)
        if not player.active:
            return SkillResult(reply="现在没有在放音乐")

        match intent.name:
            case "stop":
                self._stop()
                return SkillResult()
            case "pause":
                player.pause()
                return SkillResult()
            case "resume":
                player.resume()
                return SkillResult()
            case "next":
                return SkillResult() if player.next() else SkillResult(reply="已经是最后一首了")
            case "previous":
                return SkillResult() if player.previous() else SkillResult(reply="已经是第一首了")
            case "what":
                current = player.current
                return SkillResult(
                    reply=f"正在播放{current.title}" if current else "我也不知道这是什么歌"
                )
            case "volume_up" | "volume_down" | "volume_set":
                return self._change_volume(intent)
        return SkillResult()

    def _play(self, intent: Intent) -> SkillResult:
        """按目标类型播放。只在事件循环线程调用，不涉及音频回调。

        本地路径先重扫：AI 刚把文件写进音乐目录时，内存里的曲库还是旧的。
        URL 是现成的流，扫盘没有意义。歌名只用已有索引，用户原话没命中就交给 AI，
        避免文件还没下载就白扫一次。
        """
        assert self._player is not None
        target = str(intent.slot("query", "")).strip()
        if is_stream_url(target):
            return self._start([Track(title="在线音乐", uri=target)], "好的，播放在线音乐")
        local = as_local_path(target)
        if local is not None:
            self._library.rescan()
            if not local.is_file():
                return self._missing(local.name, intent.source)
            return self._start([Track(title=local.stem, uri=str(local))], f"好的，播放{local.stem}")

        query = clean_query(target)
        station = self._library.find_station(query) if query else None
        if station is not None:
            return self._start([station], f"好的，播放{station.title}")
        tracks = self._library.search(query)
        if not tracks:
            return self._missing(query, intent.source)
        reply = f"好的，播放{tracks[0].title}" if query else "好的，随机播放音乐"
        return self._start(tracks, reply)

    def _start(self, tracks: list[Track], reply: str) -> SkillResult:
        """替换播放列表并确保后台任务在跑。"""
        assert self._player is not None
        self._ctx.logger.info("播放：%s 等 %d 首", tracks[0].title, len(tracks))
        self._player.play(tracks)
        self._apply_volume()
        if self._task is None:
            self._task = self._ctx.spawn(self._watch(), name="音乐播放")
        return SkillResult(reply=reply)

    def _missing(self, query: str, source: str) -> SkillResult:
        """本地没有这首。用户原话交给 AI；工具调用只返回说明，避免再转交一次。

        用户路径带 ``hint``：路由器不播报，只注入当轮模型上下文，由 AI 自己开口。
        """
        if source == "agent":
            if query:
                reply = (
                    f"本地曲库没有找到{query}。"
                    "这里只查了本机文件。可以把本地文件路径或音频 URL 再传给 play_music："
                    "本地路径会先重新扫描曲库，URL 会直接播放。"
                )
            else:
                reply = "曲库是空的，请把音乐放到音乐目录里。也可以传入本地文件路径或音频 URL。"
            return SkillResult(reply=reply)
        reply = f"曲库里没有找到{query}" if query else "曲库是空的，请把音乐放到音乐目录里"
        if query:
            hint = (
                f"本地曲库没有找到{query}，请用其他工具继续查找或下载，先用一句话告诉用户你在做什么"
            )
        else:
            hint = "曲库是空的，请用其他工具继续查找或下载，先用一句话告诉用户你在做什么"
        return SkillResult(reply=reply, delegate=True, hint=hint)

    def _stop(self) -> None:
        if self._player is not None:
            self._player.stop()
        if self._task is not None:
            self._task.cancel("user")

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

    # ------------------------------------------------------------------ 后台任务与音量

    async def _watch(self) -> None:
        """后台任务「音乐播放」：等待播放列表播完；被取消时停止播放器。"""
        idle = 0
        try:
            while idle < _IDLE_POLLS:
                await asyncio.sleep(_POLL_SECONDS)
                idle = 0 if self.playing else idle + 1
        finally:
            self._task = None
            if self._player is not None and self._player.active:
                self._player.stop()

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

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="play_music",
                description="只播放本机曲库、本地文件路径和音频 URL。传入本地路径时会先重新扫描曲库；传入 URL 时不会",
                intent="play",
                args_schema=PlayMusicArgs,
            ),
            ToolSpec(name="pause_music", description="暂停音乐", intent="pause"),
            ToolSpec(name="resume_music", description="继续播放暂停的音乐", intent="resume"),
            ToolSpec(name="next_song", description="切到下一首", intent="next"),
            ToolSpec(name="stop_music", description="停止播放音乐", intent="stop"),
            ToolSpec(name="current_song", description="查询正在播放的歌曲名", intent="what"),
        ]
