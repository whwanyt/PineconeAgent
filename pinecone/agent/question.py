"""把模型给出的选项念成一句口语，并解析用户的下一句回答。

模型调用 ``ask_user`` 时用这里生成问句和 ``interrupt`` 载荷。
下一句语音或文本回到这里解析：选中、取消，或没听清。
听不清时调用方不要恢复图，保持暂停再问一遍。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pinecone.nlu.matcher import normalize

_CANCEL = {"取消", "不用了", "不用", "算了", "都不要", "不要了"}
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
_ORDINAL = re.compile(rf"(?:选|要|就|是)?第?(\d+|{'|'.join(_CN_DIGIT)})(?:个|首|项|条|号)?")


@dataclass
class Choice:
    """问句里的一个选项。``slots`` 只给解析结果用，不念给用户。"""

    id: str
    label: str
    slots: dict[str, Any] = field(default_factory=dict)


class AnswerKind(StrEnum):
    """用户对提问的回答类型。"""

    SELECT = "select"
    CANCEL = "cancel"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class Answer:
    """:func:`parse_answer` 的结果。"""

    kind: AnswerKind
    choice: Choice | None = None


@dataclass(frozen=True)
class PendingQuestion:
    """已经暂停、正在等用户回答的一次提问。"""

    prompt: str
    lines: str
    choices: list[Choice]
    limit: int


def spoken_question(choices: Sequence[Choice], *, prompt: str | None = None, limit: int = 5) -> str:
    """生成念给用户的问句。没有选项时返回空字符串。"""
    shown = _shown(choices, limit)
    if not shown:
        return ""
    body = "。".join(f"第{index}个，{choice.label}" for index, choice in enumerate(shown, 1))
    head = (prompt or "").strip().rstrip("。")
    if not head:
        head = (
            f"有{len(shown)}个结果"
            if len(choices) <= len(shown)
            else f"有{len(choices)}个结果，先说前{len(shown)}个"
        )
    return f"{head}。{body}。你要哪一个？"


def numbered_lines(choices: Sequence[Choice], *, limit: int = 5) -> str:
    """文本模式打印的编号列表。"""
    shown = _shown(choices, limit)
    return "\n".join(f"{index}. {choice.label}" for index, choice in enumerate(shown, 1))


def make_question(
    choices: Sequence[Choice], *, prompt: str | None = None, limit: int = 5
) -> PendingQuestion:
    """从选项生成一次提问。实际可选项不超过 ``limit`` 个。"""
    shown = _shown(choices, limit)
    return PendingQuestion(
        prompt=spoken_question(choices, prompt=prompt, limit=limit),
        lines=numbered_lines(choices, limit=limit),
        choices=shown,
        limit=len(shown),
    )


def to_interrupt_value(question: PendingQuestion) -> dict[str, Any]:
    """变成 ``interrupt`` 可以保存的字典。"""
    return {
        "type": "question",
        "prompt": question.prompt,
        "lines": question.lines,
        "limit": question.limit,
        "choices": [
            {"id": choice.id, "label": choice.label, "slots": choice.slots}
            for choice in question.choices
        ],
    }


def pending_from_value(value: Any) -> PendingQuestion | None:
    """从检查点里的 interrupt 值还原提问。不是提问时返回 None。"""
    if not isinstance(value, dict) or value.get("type") != "question":
        return None
    raw = value.get("choices")
    if not isinstance(raw, list):
        return None
    choices = [
        Choice(
            id=str(item.get("id", "")),
            label=str(item.get("label", "")),
            slots=dict(item.get("slots") or {}),
        )
        for item in raw
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    ]
    if not choices:
        return None
    return PendingQuestion(
        prompt=str(value.get("prompt") or ""),
        lines=str(value.get("lines") or ""),
        choices=choices,
        limit=len(choices),
    )


def parse_answer(text: str, choices: Sequence[Choice], *, limit: int = 5) -> Answer:
    """解析用户的下一句。只在念出的前 ``limit`` 项里选。"""
    shown = _shown(choices, limit)
    if not shown:
        return Answer(AnswerKind.UNCLEAR)
    normalized = normalize(text)
    if _is_cancel(normalized):
        return Answer(AnswerKind.CANCEL)
    ordinals = _ordinals(normalized)
    if len(set(ordinals)) == 1:
        index = ordinals[0]
        if 1 <= index <= len(shown):
            return Answer(AnswerKind.SELECT, shown[index - 1])
        return Answer(AnswerKind.UNCLEAR)
    if len(set(ordinals)) > 1:
        return Answer(AnswerKind.UNCLEAR)
    matched = [choice for choice in shown if _label_hit(normalized, choice.label)]
    if len(matched) == 1:
        return Answer(AnswerKind.SELECT, matched[0])
    return Answer(AnswerKind.UNCLEAR)


def _shown(choices: Sequence[Choice], limit: int) -> list[Choice]:
    if limit < 1:
        limit = 1
    return list(choices)[:limit]


def _is_cancel(normalized: str) -> bool:
    return normalized.strip("吧啊呀呢") in _CANCEL


def _ordinals(normalized: str) -> list[int]:
    if normalized.isdigit():
        return [int(normalized)]
    found: list[int] = []
    for match in _ORDINAL.finditer(normalized):
        token = match.group(1)
        number = int(token) if token.isdigit() else _CN_DIGIT[token]
        if number > 0:
            found.append(number)
    return found


def _label_hit(normalized: str, label: str) -> bool:
    name = normalize(label)
    if len(normalized) < 2 or len(name) < 2:
        return False
    return normalized == name or normalized in name or name in normalized
