"""问答接口：一次性 / SSE 流式（均需登录）。"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..schemas.ask import AskIn
from ..services import rag_service
from .deps import current_user

logger = logging.getLogger("medical-backend")

router = APIRouter(prefix="/api", tags=["ask"])


@router.post("/ask")
def ask(body: AskIn, request: Request, _user: dict = Depends(current_user)):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    system = rag_service.get_system(request.app)
    try:
        result = system.ask_with_sources(question)
    except Exception as e:  # LLM/检索异常统一转为 500，前端展示错误气泡
        logger.exception("问答失败")
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "department": result["department"],
        "related_diseases": result["related_diseases"],
        "route": result["route"],
        "disclaimer": rag_service.DISCLAIMER,
    }


@router.post("/ask/stream")
def ask_stream(body: AskIn, request: Request, _user: dict = Depends(current_user)):
    """SSE 流式问答：meta(溯源/科室/疾病) -> delta(回答文本) -> done"""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    system = rag_service.get_system(request.app)

    def sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def gen():
        try:
            for piece in system.ask_stream(question):
                if isinstance(piece, dict):
                    yield sse("meta", {**piece, "disclaimer": rag_service.DISCLAIMER})
                else:
                    yield sse("delta", {"text": piece})
            yield sse("done", {})
        except Exception as e:
            logger.exception("流式问答失败")
            yield sse("error", {"detail": str(e)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )