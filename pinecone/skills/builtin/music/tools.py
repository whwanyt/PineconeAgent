"""音乐技能交给 AI 的工具参数。

在线字段只在配置了 ``url_tool`` 或 ``list_tools`` 时出现，避免模型在没有音乐服务时
去填歌单参数。``songs`` 只放用户选中的一首；整张歌单、榜单和歌手搜索走 ``list_tool``。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from pinecone.skills import ToolSpec

from .online import SongArg
from .playqueue import PlayMode

_QUERY = "歌名、歌手、电台名、本地文件路径或音频 URL。歌名为空表示随机播放本地曲库"
_SONGS = (
    "只放用户选中的那一首。在线歌填写 id、source、name、artist，有 extra 就带上。"
    "不要自己取播放地址，也不要把一整份搜索结果抄进来"
)
_LIST_TOOL = "整张歌单、专辑、榜单、平台热歌，或搜索一批歌时使用。只能填配置里允许的工具名"
_LIST_ARGS = (
    "取歌参数。search_songs 传 keyword；歌单、专辑、榜单传 source 和 id；"
    "平台热歌传 source。没有 limit 时按 50 首"
)


class SetPlayModeArgs(BaseModel):
    """切换播放模式。"""

    mode: PlayMode = Field(description="顺序播放、列表循环、单曲循环或随机播放")


class EditQueueArgs(BaseModel):
    """跳转、删除或清空当前播放列表。"""

    action: Literal["jump", "remove", "clear"] = Field(
        description="jump 跳到某一首，remove 删掉某一首，clear 只留下正在放的"
    )
    position: int = Field(default=0, description="从 1 开始的位置。0 表示当前这首。clear 时忽略")


class ListPlaylistsArgs(BaseModel):
    """查看歌单。"""

    name: str = Field(default="", description="空着列出全部歌单；给出名字则列出该歌单里的歌")


class RemovePlaylistArgs(BaseModel):
    """从歌单里删一首。"""

    name: str = Field(description="歌单名。收藏、红心、我喜欢都是「我喜欢」")
    position: int = Field(default=0, description="从 1 开始的位置。0 表示按 query 找")
    query: str = Field(default="", description="要删的歌名。position 不是 0 时忽略")


class DeletePlaylistArgs(BaseModel):
    """删除整个歌单。调用前先向用户确认。我喜欢不能删。"""

    name: str = Field(description="歌单名")


def music_tools(*, url_tool: str, list_tools: list[str]) -> list[ToolSpec]:
    """按是否接了音乐服务，决定播放工具要不要在线字段。"""
    online = bool(url_tool or list_tools)
    names = "、".join(list_tools)
    play_args = _source_model(
        "PlayMusicArgs",
        online=online,
        list_names=names,
    )
    queue_args = _source_model(
        "QueueMusicArgs",
        online=online,
        list_names=names,
        extra={
            "next": (
                bool,
                Field(default=False, description="为真时插到下一首，否则加到播放列表末尾"),
            )
        },
    )
    add_args = _source_model(
        "AddPlaylistArgs",
        online=online,
        list_names=names,
        extra={
            "name": (str, Field(description="歌单名。没有这个歌单就新建。收藏类名字会归到我喜欢")),
            "whole_queue": (
                bool,
                Field(default=False, description="为真时把当前播放列表保存进这个歌单"),
            ),
        },
    )
    return [
        ToolSpec(
            name="play_music",
            description=(
                "替换播放列表并开始播放。本地歌名、路径、电台用 query。"
                "自建歌单用 playlist。整张在线歌单、专辑、榜单、平台热歌或一批搜索结果用 list_tool，"
                "不要把每首歌抄进 songs，也不要自己取播放地址。songs 只放用户选中的那一首。"
                "优先级是 list_tool、songs、playlist、query"
            ),
            intent="play",
            args_schema=play_args,
        ),
        ToolSpec(
            name="queue_music",
            description="把歌加入当前播放列表。next 为真时作为下一首。没有在播时直接开始放",
            intent="enqueue",
            args_schema=queue_args,
        ),
        ToolSpec(name="pause_music", description="暂停音乐", intent="pause"),
        ToolSpec(name="resume_music", description="继续播放暂停的音乐", intent="resume"),
        ToolSpec(name="next_song", description="切到下一首", intent="next"),
        ToolSpec(name="previous_song", description="切到上一首", intent="previous"),
        ToolSpec(name="stop_music", description="停止播放音乐", intent="stop"),
        ToolSpec(
            name="current_song",
            description="查询正在播放的歌，以及播放列表里接下来的歌",
            intent="what",
        ),
        ToolSpec(
            name="set_play_mode",
            description="切换顺序播放、列表循环、单曲循环或随机播放",
            intent="mode",
            args_schema=SetPlayModeArgs,
        ),
        ToolSpec(
            name="edit_queue",
            description="跳到播放列表的某一首、删掉某一首，或只留下正在放的",
            intent="edit",
            args_schema=EditQueueArgs,
        ),
        ToolSpec(
            name="list_playlists",
            description="列出歌单，或列出某个歌单里的歌",
            intent="list_playlists",
            args_schema=ListPlaylistsArgs,
        ),
        ToolSpec(
            name="add_to_playlist",
            description=(
                "把歌加入自建歌单。什么都不指定时加入当前这首。收藏进我喜欢。"
                "歌单不存在就新建。whole_queue 把当前播放列表放进去"
            ),
            intent="add_playlist",
            args_schema=add_args,
        ),
        ToolSpec(
            name="remove_from_playlist",
            description="从歌单里删除一首",
            intent="remove_playlist_track",
            args_schema=RemovePlaylistArgs,
        ),
        ToolSpec(
            name="delete_playlist",
            description="删除整个歌单。调用前先向用户确认。我喜欢不能删除",
            intent="delete_playlist",
            args_schema=DeletePlaylistArgs,
        ),
    ]


def _source_model(
    name: str,
    *,
    online: bool,
    list_names: str,
    extra: dict[str, Any] | None = None,
) -> type[BaseModel]:
    """播放和加歌共用的歌源参数。没接音乐服务时不放在线字段。"""
    fields: dict[str, Any] = {
        "query": (str, Field(default="", description=_QUERY)),
        "playlist": (str, Field(default="", description="自建歌单名")),
    }
    if online:
        fields["songs"] = (list[SongArg], Field(default_factory=list, description=_SONGS))
        fields["list_tool"] = (
            str,
            Field(default="", description=f"{_LIST_TOOL}。可选：{list_names or '无'}"),
        )
        fields["list_args"] = (dict[str, Any], Field(default_factory=dict, description=_LIST_ARGS))
    if extra:
        fields.update(extra)
    return create_model(name, **fields)
