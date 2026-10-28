"""右侧播放列表：显示当前队列，上下选择，回车从那一首开始放。

界面只根据快照和 :class:`~pinecone.core.events.PlaylistUpdated` 重画，
切歌命令交给会话门面，不经过对话。
"""

from __future__ import annotations

from textual.widgets import ListItem, ListView, Static


class TrackRow(ListItem):
    """播放列表里的一行。``track_index`` 从 1 数。"""

    def __init__(self, index: int, title: str, *, current: bool) -> None:
        mark = "▶" if current else " "
        super().__init__(Static(f"{mark} {index}. {title}"))
        self.track_index = index
        self.track_title = title
        self.current = current


class PlaylistView(ListView):
    """当前播放队列。

    焦点不在这里时，新的队列把高亮跟到正在放的那首。
    焦点在这里选歌时保留光标，免得自动切歌把用户正在看的那一行拽走。
    回车由外层界面换成 ``select_track``，本控件不自己播歌。

    只在界面的事件循环线程里更新。
    """

    def __init__(self) -> None:
        super().__init__(initial_index=None, id="playlist")

    async def sync(self, tracks: tuple[str, ...], current: int) -> None:
        """按最新队列重画。

        Args:
            tracks: 按播放顺序的曲目名。
            current: 正在播放的序号，从 1 数；没有则为 0。
        """
        # 先记下光标：clear 会把高亮清掉
        focused = self.has_focus
        cursor = self.index
        await self.clear()
        if not tracks:
            return
        await self.extend(
            TrackRow(index, title, current=index == current)
            for index, title in enumerate(tracks, 1)
        )
        if focused and cursor is not None:
            self.index = min(cursor, len(tracks) - 1)
        elif current > 0:
            self.index = current - 1
