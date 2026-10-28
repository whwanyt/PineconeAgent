"""模型用来向用户提问的工具。

工具返回多个候选时，由模型自己调用 ``ask_user``，不要在技能或 MCP 结果里自动暂停。
这个工具内部调用 LangGraph ``interrupt``：问句念给用户，下一句原话成为工具结果，
模型再决定播放、删除或继续查询。

恢复时工具函数会从头执行，参数仍是模型当初传入的问题和选项，所以不必另做缓存。
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from pinecone.agent.question import (
    AnswerKind,
    Choice,
    make_question,
    parse_answer,
    to_interrupt_value,
)

logger = logging.getLogger(__name__)

_CANCELLED = "用户取消了选择。不要继续这项操作，除非用户重新提出。"


class AskUserArgs(BaseModel):
    """``ask_user`` 的参数。选项会按顺序念给用户。"""

    question: str = Field(
        description="你在调用前已经对用户说的那句问话，本工具不会再念一遍，只念下面的选项"
    )
    options: list[str] = Field(
        description="要念给用户的短选项，按顺序，例如「宝船，DOUDOU」「宝船，燕云十六声」"
    )


def ask_user_tool(choice_limit: int = 5) -> BaseTool:
    """构造提问工具。``choice_limit`` 是一次最多念几个选项。"""

    async def ask_user(question: str, options: list[str]) -> str:
        labels = [item.strip() for item in options if item and item.strip()]
        if len(labels) < 2:
            return "选项不足两个，不必问用户，请直接处理。"
        choices = [
            Choice(id=str(index), label=label, slots={"index": index})
            for index, label in enumerate(labels)
        ]
        pending = make_question(choices, prompt=question, limit=choice_limit, include_head=False)
        # 第一次在这里暂停；恢复后返回值是用户原话
        answer = interrupt(to_interrupt_value(pending))
        parsed = parse_answer(str(answer), pending.choices, limit=pending.limit)
        if parsed.kind is AnswerKind.CANCEL:
            logger.info("用户取消了提问")
            return _CANCELLED
        if parsed.kind is AnswerKind.SELECT and parsed.choice is not None:
            number = int(parsed.choice.slots["index"]) + 1
            logger.info("用户选中第 %s 个：%s", number, parsed.choice.label)
            return (
                f"用户选择了第{number}个：{parsed.choice.label}。"
                "请只处理这一项，不要再问，也不要改选其他项。"
            )
        spoken = str(answer).strip()
        logger.info("用户没有按序号选择：%s", spoken)
        return f"用户没有选第几个，原话是：{spoken}。请根据这句话继续，不要再把同一批选项念一遍。"

    return StructuredTool.from_function(
        coroutine=ask_user,
        name="ask_user",
        description=(
            "有多个候选、用户又没指定是哪一个时调用。先自己用一句话问用户，再调用本工具。"
            "本工具只念「第几个」和选项，不会重复你刚说的那句。"
            "在这个工具返回之前，不要自己选第一条，也不要播放、删除或再次查询。"
        ),
        args_schema=AskUserArgs,
    )
