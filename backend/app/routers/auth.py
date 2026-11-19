"""认证接口：注册 / 登录 / 当前用户。"""

from fastapi import APIRouter, Depends

from ..schemas.auth import AuthIn
from ..services import auth_service
from .deps import current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register")
def register(body: AuthIn):
    return auth_service.register(body)


@router.post("/login")
def login(body: AuthIn):
    return auth_service.login(body)


@router.get("/me")
def me(user: dict = Depends(current_user)):
    return user