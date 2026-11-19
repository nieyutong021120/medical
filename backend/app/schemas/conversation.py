"""会话/消息请求模型。"""

from pydantic import BaseModel


class ConversationIn(BaseModel):
    title: str = "新对话"
    createdAt: int | None = None
    messages: list[dict] | None = None


class ConversationPatch(BaseModel):
    title: str


class MessagePatch(BaseModel):
    feedback: str | None = None