"""路由公共依赖：从 Authorization 头解析当前用户。"""

from fastapi import Header, HTTPException

from ..db import database as db
from ..services import auth_service


def current_user(authorization: str | None = Header(default=None)) -> dict:
    """从 Authorization: Bearer <JWT> 解析用户身份；缺失/无效/过期一律 401。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    user_id = auth_service.decode_token(authorization[7:])
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user