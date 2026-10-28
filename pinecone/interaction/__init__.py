"""交互门面。

TUI 和以后的 HTTP 接口只通过 :class:`Session` 驱动核心，并订阅事件总线上的事件。
核心（对话、技能、语音）不依赖本包。
"""

from pinecone.interaction.session import Frontend, Playback, PlaylistSnapshot, Session

__all__ = ["Frontend", "Playback", "PlaylistSnapshot", "Session"]
