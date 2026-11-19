"""应用组装：创建 app、挂中间件、注册路由、管理 RAG 生命周期。业务逻辑不在此文件。

运行（all-in-rag conda 环境，仓库根目录）：
  D:/miniconda/envs/all-in-rag/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging import setup_logging
from .db.database import init_db
from .routers import ask, auth, conversations, health
from .services import rag_service

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    rag_service.initialize(app)
    yield
    rag_service.shutdown(app)


app = FastAPI(title="Medical Assistant RAG API", lifespan=lifespan)
# 旧：allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"]（写死前端来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(ask.router)
app.include_router(conversations.router)