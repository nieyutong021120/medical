"""健康检查（公开，不要求登录）。"""

from fastapi import APIRouter, Request

from ..services import rag_service

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health(request: Request):
    rag_service.get_system(request.app)
    return {"ok": True, "stats": request.app.state.stats}