"""全局配置：环境加载 + 配置项（唯一来源）。

加载顺序与旧 backend/main.py 一致：
medical_rag/.env（override=True，RAG 密钥唯一权威）→ backend/.env（部署配置）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]  # backend/
ROOT = BACKEND_DIR.parent                          # 仓库根
RAG_DIR = ROOT / "medical_rag"

# 嵌入模型已在本机 HF 缓存，离线加载避免启动时访问网络
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 以 RAG 的 .env 为唯一权威来源：override=True 强制覆盖系统/会话里残留的旧 MOONSHOT_API_KEY
load_dotenv(RAG_DIR / ".env", override=True)
# 后端部署配置（FRONTEND_URL / JWT_SECRET）来自 backend/.env；不覆盖已存在的系统环境变量
load_dotenv(BACKEND_DIR / ".env")


class Settings:
    """后端配置项；JWT_SECRET 缺失时启动即报错，避免签发不可验证的 token。"""

    JWT_SECRET: str = os.environ.get("JWT_SECRET", "")
    JWT_ALGORITHM: str = "HS256"
    JWT_TTL_DAYS: int = 7           #JWT 7天有效期
    FRONTEND_URL: str = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    DATA_PATH: Path = ROOT / "Huatuo-Lite" / "format_data.jsonl"
    INDEX_PATH: Path = RAG_DIR / "vector_index"


settings = Settings()
if not settings.JWT_SECRET:
    raise RuntimeError("缺少 JWT_SECRET：请在 backend/.env 配置（参考 backend/.env.example）")