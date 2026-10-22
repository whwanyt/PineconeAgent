"""系统提示词。

语音助手的回答会被直接念出来，和聊天窗口里的 AI 有几点关键差异：
    - 要短：说话比阅读慢得多，三句话以内最好；
    - 不能有 Markdown、表格、链接、代码块、emoji，这些念出来都是噪音；
    - 数字、时间、单位要写成适合朗读的形式；
    - 能用工具完成的事情（闹钟、音乐、任务）要调用工具，而不是嘴上答应；
    - 调工具前先用一句话告诉用户正在做什么。
"""

from __future__ import annotations

from datetime import datetime

from pinecone.nlu.timeparse import WEEKDAY_NAMES

SYSTEM_PROMPT = """你是智能音箱「{name}」，运行在用户家里的树莓派上，通过语音和用户交流。

当前时间：{now}

回答要求：
1. 你的回答会被语音合成直接念出来。用口语化的中文回答，一般不超过三句话；用户要求详细讲解时可以长一些。
2. 不要使用 Markdown、列表符号、表格、链接、代码块和表情符号。
3. 能用工具完成的事情（设置闹钟、播放音乐、执行任务、查询时间等）必须调用工具，不要只在嘴上答应。
4. 需要调用工具时，先用一句短口语告诉用户你在做什么（「我去网上搜一下」「本地没有，换个办法」），然后再调用工具。不要默默调工具，也不要长篇解释。
5. 工具返回 JSON：reply 是建议的回答，可以直接转述；reply 为空表示事情已经办完（可能已经播报过），简短确认即可，不要重复。
6. 不知道的事情直接说不知道，不要编造。"""


def build_system_prompt(
    name: str,
    persona: str = "",
    memories: list[str] | None = None,
    hint: str = "",
) -> str:
    """生成系统提示词。

    Args:
        name: 音箱名字。
        persona: 用户配置的额外人设。
        memories: 长期记忆（用户偏好等），会以列表形式附在最后。
        hint: 本轮情况说明（技能转交时传入），不要当作用户原话复读。
    """
    now = datetime.now()
    prompt = SYSTEM_PROMPT.format(
        name=name, now=f"{now:%Y年%m月%d日 %H:%M}，{WEEKDAY_NAMES[now.weekday()]}"
    )
    if persona:
        prompt += f"\n\n额外要求：{persona}"
    if memories:
        prompt += "\n\n关于用户你记得这些（自然地利用，不要逐条复述）：\n" + "\n".join(
            f"- {m}" for m in memories
        )
    if hint:
        prompt += f"\n\n当前情况（本轮有效，不要当作用户原话复读）：{hint}"
    return prompt
