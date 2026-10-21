"""朗读前的文本处理：分句、清洗 Markdown 符号。

为什么要分句？
    大模型是逐字流式输出的。如果等整段回复生成完再合成，用户要等好几秒才听到声音。
    按句切分后，第一句生成完就开始合成播放，后续句子边生成边合成，首字延迟大幅降低。
"""

from __future__ import annotations

import re

# 句末标点：遇到即切分
_SENTENCE_END = set("。！？!?；;\n")
# 次级标点：句子过长时在这里切分，避免第一句太长导致首字延迟高
_SOFT_BREAK = set("，,、：:")
# 超过该长度后，遇到次级标点也切分
_SOFT_LIMIT = 24

# 朗读时需要去掉的 Markdown / 特殊符号
_MARKDOWN = re.compile(r"[*#`>_~|]+|\[(.*?)\]\(.*?\)")
_SPACES = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """清洗文本：去掉 Markdown 符号（保留链接文字）、合并空白。"""
    text = _MARKDOWN.sub(lambda m: m.group(1) or "", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)  # 列表符号
    return _SPACES.sub(" ", text).strip()


def _speakable(text: str) -> bool:
    """是否包含可朗读的字符（字母、数字或汉字），纯标点不需要合成。"""
    return any(ch.isalnum() for ch in text)


class SentenceSplitter:
    """流式分句器：不断 :meth:`feed` 文本片段，返回已经完整的句子。"""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, piece: str) -> list[str]:
        """追加文本片段，返回本次产生的完整句子（已清洗、非空）。"""
        sentences: list[str] = []
        for ch in piece:
            self._buffer += ch
            if ch in _SENTENCE_END or (ch in _SOFT_BREAK and len(self._buffer) >= _SOFT_LIMIT):
                sentences.extend(self._pop())
        return sentences

    def flush(self) -> list[str]:
        """取出缓冲区剩余内容（流结束时调用）。"""
        return self._pop()

    def _pop(self) -> list[str]:
        text, self._buffer = clean_for_speech(self._buffer), ""
        return [text] if _speakable(text) else []


def split_sentences(text: str) -> list[str]:
    """把完整文本切分成句子列表。"""
    splitter = SentenceSplitter()
    return splitter.feed(text) + splitter.flush()
