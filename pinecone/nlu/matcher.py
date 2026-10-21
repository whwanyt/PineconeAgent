"""文本归一化。

识别结果会带有标点、空格、全角字符，例如 “现在几点了？”、“播放 周杰伦 的歌。”。
规则匹配前统一做归一化，技能的正则就不需要到处考虑标点和空格：

    “现在几点了？” → “现在几点了”
    “播放 周杰伦 的歌。” → “播放周杰伦的歌”
    “Play Music” → “play music”（英文单词之间的空格保留）
"""

from __future__ import annotations

import re
import unicodedata

# 中文语境下需要去掉的标点（含全角 / 半角）
_PUNCT = re.compile(r"[，。！？、；：“”‘’（）《》【】…—,.!?;:\"'()\[\]<>~·]")
# 两个汉字（或汉字与字母数字）之间的空白去掉，两个英文单词之间保留
_CJK_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])\s+|\s+(?=[\u4e00-\u9fff])")


def normalize(text: str) -> str:
    """归一化用户文本：全角转半角、去标点、去中文周围空白、英文转小写。"""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT.sub("", text)
    text = _CJK_SPACE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()
