"""认证业务：注册/登录校验、JWT 签发与解析。密码哈希等 SQL 细节在 db/database.py。"""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException

from ..core.config import settings
from ..db import database as db
from ..schemas.auth import AuthIn


def _make_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": user_id, "iat": now, "exp": now + timedelta(days=settings.JWT_TTL_DAYS)},   # 7天过期声明
        settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM,
    )


def decode_token(token: str) -> str:
    """解析 JWT 返回 user_id；无效/过期抛 401。"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="登录凭证无效")
    return payload.get("sub", "")


def register(body: AuthIn) -> dict:
    username = body.username.strip()
    if not 2 <= len(username) <= 32:
        raise HTTPException(status_code=400, detail="用户名需为 2-32 个字符")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    user = db.create_user(username, body.password)
    if not user:
        raise HTTPException(status_code=409, detail="用户名已存在")
    return {"token": _make_token(user["id"]), "user": user}


def login(body: AuthIn) -> dict:
    user = db.get_user_by_name(body.username.strip())
    if not user or not db.verify_password(user, body.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {
        "token": _make_token(user["id"]),
        "user": {"id": user["id"], "username": user["username"]},
    }