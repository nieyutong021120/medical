"""问答请求模型。"""

from pydantic import BaseModel


class AskIn(BaseModel):
    question: str