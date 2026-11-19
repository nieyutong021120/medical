"""FastAPI 后端：包装 medical_rag，向前端暴露 /api/health、/api/ask、/api/ask/stream。

运行（all-in-rag conda 环境，仓库根目录）：
  D:/miniconda/envs/all-in-rag/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

import importlib.util
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = ROOT / "medical_rag"

# 嵌入模型已在本机 HF 缓存，离线加载避免启动时访问网络
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 以 RAG 的 .env 为唯一权威来源：override=True 强制覆盖系统/会话里残留的旧 MOONSHOT_API_KEY
load_dotenv(RAG_DIR / ".env", override=True)
# 后端部署配置（FRONTEND_URL）来自 backend/.env；不覆盖已存在的系统环境变量
load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(RAG_DIR))
# 让 db 在 `uvicorn backend.main:app` 与 `fastapi dev backend/main.py` 两种启动方式下都可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

# JWT 认证配置：密钥来自 backend/.env（见 backend/.env.example）
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("缺少 JWT_SECRET：请在 backend/.env 配置（参考 backend/.env.example）")
JWT_ALGORITHM = "HS256"
JWT_TTL = timedelta(days=7)

logger = logging.getLogger("medical-backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _load_rag_module():
    spec = importlib.util.spec_from_file_location("medical_rag_main", RAG_DIR / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rag = _load_rag_module()
import kb_modules.generation as rag_generation  # noqa: E402  (sys.path 已含 RAG_DIR)

DISCLAIMER = rag_generation.DISCLAIMER.strip().lstrip("-").strip()


class AskIn(BaseModel):
    question: str


class ConversationIn(BaseModel):
    title: str = "新对话"
    createdAt: int | None = None
    messages: list[dict] | None = None


class ConversationPatch(BaseModel):
    title: str


class MessagePatch(BaseModel):
    feedback: str | None = None


class AuthIn(BaseModel):
    username: str
    password: str


def uid() -> str:
    """持久化 id（会话/消息）统一由后端生成，前端不再自带 id。"""
    return str(uuid.uuid4())


# 旧：Cookie 匿名身份（client_id httponly Cookie，30 天）；现改 JWT 用户登录认证。
# def client_id(request: Request, response: Response) -> str:
#     """从 httponly Cookie 读取浏览器身份；首次访问由后端生成并下发，30 天有效。"""
#     cid = request.cookies.get("client_id")
#     if not cid:
#         # 旧：cid = uuid.uuid4().hex（32位无连字符）；统一为规范36位，与 uid() 主键风格一致
#         cid = str(uuid.uuid4())
#         response.set_cookie(
#             "client_id", cid,
#             httponly=True, samesite="lax",
#             max_age=60 * 60 * 24 * 30,
#         )
#     return cid


def _make_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {"sub": user_id, "iat": now, "exp": now + JWT_TTL},
        JWT_SECRET, algorithm=JWT_ALGORITHM,
    )


def current_user(authorization: str | None = Header(default=None)) -> dict:
    """从 Authorization: Bearer <JWT> 解析用户身份；缺失/无效/过期一律 401。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    try:
        payload = jwt.decode(authorization[7:], JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="登录凭证无效")
    user = db.get_user_by_id(payload.get("sub", ""))
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # 旧：RAG_VARIANT 双版本路径（已弃用）；数据文件由 data/filter_score5.py 原位替换，此处读标准名
    # config = rag.MedicalConfig(
    #     data_path=str(_data_path),
    #     index_save_path=str(_index_path),
    # )
    # logger.info("加载数据版本: %s (%s)", _rag_variant, _data_path.name)
    config = rag.MedicalConfig(
        data_path=str(ROOT / "Huatuo-Lite" / "format_data.jsonl"),
        index_save_path=str(RAG_DIR / "vector_index"),
    )
    system = rag.MedicalSystem(config)
    system.initialize_system()
    system.build_knowledge_base()
    app.state.system = system
    app.state.stats = system.ingestion.get_statistics()
    logger.info("医疗 RAG 后端就绪")
    yield
    app.state.system = None


app = FastAPI(title="Medical Assistant RAG API", lifespan=lifespan)
# 旧：allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"]（写死前端来源）
_frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
# 旧：RAG_VARIANT 环境变量双版本选择（已弃用）；现由 data/filter_score5.py 原位替换数据文件，
# 后端与 build_index 读同一标准文件名，无需任何开关。
# _RAG_VARIANTS = {
#     "full": (ROOT / "Huatuo-Lite" / "format_data.jsonl", RAG_DIR / "vector_index"),
#     "score5": (ROOT / "Huatuo-Lite" / "format_data_score5.jsonl", RAG_DIR / "vector_index_score5"),
# }
# _rag_variant = os.environ.get("RAG_VARIANT", "full")
# _data_path, _index_path = _RAG_VARIANTS.get(_rag_variant, _RAG_VARIANTS["full"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_system():
    system = getattr(app.state, "system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="RAG 尚未就绪")
    return system


@app.get("/api/health")
def health():
    _get_system()
    return {"ok": True, "stats": app.state.stats}


@app.post("/api/auth/register")
def register(body: AuthIn):
    username = body.username.strip()
    if not 2 <= len(username) <= 32:
        raise HTTPException(status_code=400, detail="用户名需为 2-32 个字符")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    user = db.create_user(username, body.password)
    if not user:
        raise HTTPException(status_code=409, detail="用户名已存在")
    return {"token": _make_token(user["id"]), "user": user}


@app.post("/api/auth/login")
def login(body: AuthIn):
    user = db.get_user_by_name(body.username.strip())
    if not user or not db.verify_password(user, body.password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return {
        "token": _make_token(user["id"]),
        "user": {"id": user["id"], "username": user["username"]},
    }


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    return user


@app.post("/api/ask")
def ask(body: AskIn, _user: dict = Depends(current_user)):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    system = _get_system()
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
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/ask/stream")
def ask_stream(body: AskIn, _user: dict = Depends(current_user)):
    """SSE 流式问答：meta(溯源/科室/疾病) -> delta(回答文本) -> done"""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    system = _get_system()

    def sse(event: str, data) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def gen():
        try:
            for piece in system.ask_stream(question):
                if isinstance(piece, dict):
                    yield sse("meta", {**piece, "disclaimer": DISCLAIMER})
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


@app.get("/api/conversations")
def get_conversations(user: dict = Depends(current_user)):
    # 旧：cid: str = Depends(client_id)（Cookie 匿名身份）
    return db.list_conversations(user["id"])


@app.post("/api/conversations")
def create_conversation(body: ConversationIn, user: dict = Depends(current_user)):
    conv_id = uid()
    # 撤销删除会带回旧消息，其 id 同样由后端重新生成
    messages = [{**m, "id": uid()} for m in body.messages] if body.messages else None
    return db.create_conversation(user["id"], conv_id, body.title, body.createdAt, messages)


@app.patch("/api/conversations/{conv_id}")
def patch_conversation(conv_id: str, body: ConversationPatch, user: dict = Depends(current_user)):
    if not db.update_conversation(user["id"], conv_id, body.title):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
def remove_conversation(conv_id: str, user: dict = Depends(current_user)):
    if not db.delete_conversation(user["id"], conv_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@app.post("/api/conversations/{conv_id}/messages")
def add_message(conv_id: str, message: dict = Body(...), user: dict = Depends(current_user)):
    if not message.get("role"):
        raise HTTPException(status_code=400, detail="message 需含 role")
    message["id"] = uid()
    if not db.append_message(user["id"], conv_id, message):
        raise HTTPException(status_code=404, detail="会话不存在")
    return message


@app.patch("/api/messages/{msg_id}")
def patch_message(msg_id: str, body: MessagePatch, user: dict = Depends(current_user)):
    if not db.update_message_feedback(user["id"], msg_id, body.feedback):
        raise HTTPException(status_code=404, detail="消息不存在")
    return {"ok": True}