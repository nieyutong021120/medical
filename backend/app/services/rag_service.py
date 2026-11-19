"""RAG 系统引导与访问：加载 medical_rag、生命周期初始化、就绪检查。

medical_rag 包本身保持独立不动，这里沿用旧 backend/main.py 的 importlib 引导方式。
"""

import importlib.util
import logging
import sys

from fastapi import FastAPI, HTTPException

from ..core.config import RAG_DIR, settings

logger = logging.getLogger("medical-backend")

# 让 medical_rag 内部的 kb_modules 等可导入
sys.path.insert(0, str(RAG_DIR))


def _load_rag_module():
    spec = importlib.util.spec_from_file_location("medical_rag_main", RAG_DIR / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rag = _load_rag_module()
import kb_modules.generation as rag_generation  # noqa: E402  (sys.path 已含 RAG_DIR)

DISCLAIMER = rag_generation.DISCLAIMER.strip().lstrip("-").strip()


def initialize(app: FastAPI) -> None:
    """lifespan 启动段：建索引/加载知识库，挂到 app.state。"""
    config = rag.MedicalConfig(
        data_path=str(settings.DATA_PATH),
        index_save_path=str(settings.INDEX_PATH),
    )
    system = rag.MedicalSystem(config)
    system.initialize_system()
    system.build_knowledge_base()
    app.state.system = system
    app.state.stats = system.ingestion.get_statistics()
    logger.info("医疗 RAG 后端就绪")


def shutdown(app: FastAPI) -> None:
    app.state.system = None


def get_system(app: FastAPI):
    system = getattr(app.state, "system", None)
    if system is None:
        raise HTTPException(status_code=503, detail="RAG 尚未就绪")
    return system