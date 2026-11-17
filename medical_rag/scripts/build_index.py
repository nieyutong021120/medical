"""离线构建 FAISS + BM25 索引（17.8万条QA，CPU编码耗时较长，建一次即可）

运行（仓库根目录）：
  D:/miniconda/envs/all-in-rag/python.exe medical_rag/scripts/build_index.py
"""

import os
import sys
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))

# 嵌入模型已在本机HF缓存，离线加载避免访问网络
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv(RAG_DIR / ".env", override=True)

from config import MedicalConfig
from kb_modules import IngestionModule, IndexingModule


def main():
    config = MedicalConfig(
        data_path=str(RAG_DIR.parent / "Huatuo-Lite" / "format_data.jsonl"),
        index_save_path=str(RAG_DIR / "vector_index"),
    )
    ingestion = IngestionModule(config.data_path)
    ingestion.load_documents()

    indexing = IndexingModule(
        model_name=config.embedding_model,
        index_save_path=config.index_save_path,
    )
    indexing.build_vector_index(ingestion.documents)
    indexing.build_bm25_index(ingestion.get_bm25_documents(), k=config.retriever_k)
    indexing.save_index()

    # 验证落盘索引可回载
    indexing2 = IndexingModule(
        model_name=config.embedding_model,
        index_save_path=config.index_save_path,
    )
    assert indexing2.load_index() is not None, "FAISS 回载失败"
    assert indexing2.load_bm25_index() is not None, "BM25 回载失败"
    hits = indexing2.vectorstore.similarity_search("高血压患者能吃党参吗", k=3)
    print("\n回载验证检索样例:")
    for h in hits:
        print(" -", h.page_content[:50], "|", h.metadata["label"], "| score", h.metadata["score"])
    print("\n索引构建并验证完成:", config.index_save_path)


if __name__ == "__main__":
    main()