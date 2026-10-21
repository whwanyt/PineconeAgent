"""助手状态机。

一次完整的语音交互会经历以下状态::

    IDLE ──唤醒──▶ LISTENING ──说完──▶ THINKING ──有回复──▶ SPEAKING ──播完──▶ FOLLOW_UP
     ▲                │                  │                    │                   │
     └──────超时/无内容─┴─────────────────┴────────────────────┴──追问窗口超时──────┘

    打断：THINKING / SPEAKING 状态下检测到唤醒词 ──▶ LISTENING

- ``FOLLOW_UP``：播报结束后的追问窗口（默认约 6 秒），用户无需再说唤醒词即可继续对话；
- 闹钟响铃、自定义任务的主动播报允许从 ``IDLE`` 直接进入 ``SPEAKING``；
- 任何状态都可以回到 ``IDLE``（出错、取消、超时时用于复位）。

状态机只负责“记录当前状态 + 校验迁移是否合法 + 发布 StateChanged 事件”，
不包含任何业务逻辑，业务逻辑由订阅 StateChanged 的模块各自实现。
"""

from __future__ import annotations

import logging
from enum import StrEnum

from pinecone.core.bus import EventBus
from pinecone.core.events import StateChanged

logger = logging.getLogger(__name__)


class AssistantState(StrEnum):
    """助手所处的交互状态。"""

    IDLE = "idle"
    """待机，只运行唤醒词检测。"""
    LISTENING = "listening"
    """已唤醒，正在录制用户说的话。"""
    THINKING = "thinking"
    """正在识别 / 路由 / 调用技能或大模型。"""
    SPEAKING = "speaking"
    """正在播报回复。"""
    FOLLOW_UP = "follow_up"
    """追问窗口：不需要唤醒词，直接说话即可继续对话。"""


S = AssistantState

# 合法的状态迁移表（不含“任意状态 -> IDLE”，那条规则在代码里统一放行）
_TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    S.IDLE: {S.LISTENING, S.SPEAKING},
    S.LISTENING: {S.THINKING},
    S.THINKING: {S.SPEAKING, S.LISTENING, S.FOLLOW_UP},
    S.SPEAKING: {S.FOLLOW_UP, S.LISTENING, S.SPEAKING},
    S.FOLLOW_UP: {S.LISTENING, S.SPEAKING},
}


class InvalidTransition(RuntimeError):
    """非法的状态迁移。"""


class StateMachine:
    """记录并校验助手状态。

    非线程安全：只能在事件循环线程中调用。
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._state = AssistantState.IDLE

    @property
    def state(self) -> AssistantState:
        """当前状态。"""
        return self._state

    def can_transition(self, new: AssistantState) -> bool:
        """判断能否迁移到 ``new``。"""
        return new == AssistantState.IDLE or new in _TRANSITIONS[self._state]

    async def transition(self, new: AssistantState, reason: str = "") -> None:
        """迁移到新状态并发布 :class:`StateChanged` 事件。

        迁移到与当前相同的状态（SPEAKING -> SPEAKING 除外，表示开始播报下一段）时什么都不做。

        Args:
            new: 目标状态。
            reason: 迁移原因，便于日志排查。

        Raises:
            InvalidTransition: 迁移不合法。
        """
        old = self._state
        if new == old and new != AssistantState.SPEAKING:
            return
        if not self.can_transition(new):
            raise InvalidTransition(f"非法状态迁移：{old} -> {new}（{reason}）")
        self._state = new
        logger.info("状态：%s -> %s %s", old, new, f"（{reason}）" if reason else "")
        await self._bus.publish(StateChanged(old=old.value, new=new.value, reason=reason))

    def is_busy(self) -> bool:
        """是否处于“前台忙碌”状态（思考或播报），这两个状态允许被打断。"""
        return self._state in (AssistantState.THINKING, AssistantState.SPEAKING)
