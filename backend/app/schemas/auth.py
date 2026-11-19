"""认证相关请求/响应模型。"""

from pydantic import BaseModel


class AuthIn(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: str
    username: str


class TokenOut(BaseModel):
    token: str
    user: UserOut