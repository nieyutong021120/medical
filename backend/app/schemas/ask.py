"""问答请求模型。"""

from pydantic import BaseModel


class AskIn(BaseModel):
    question: str
    # 多轮对话：可选会话 id，后端据此加载本人历史（不属于该用户则视为无历史）
    conversation_id: str | None = None