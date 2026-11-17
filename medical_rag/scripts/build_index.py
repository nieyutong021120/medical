"""离线构建 FAISS + BM25 索引（CPU编码耗时较长，建一次即可）

运行（仓库根目录）：
  python medical_rag/scripts/build_index.py
始终读 Huatuo-Lite/format_data.jsonl、写 medical_rag/vector_index。
低内存服务器流程：data/download_data.py 下载全量 -> data/filter_score5.py 原位过滤为
score=5 -> 重跑本脚本重建索引（此时 format_data.jsonl 已是 score5 子集）。
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
    # 旧：位置参数选择数据/索引目录（已弃用）；现数据文件由 data/filter_score5.py 原位替换，
    # 这里始终读标准名 format_data.jsonl / 写 vector_index，服务器过滤后直接重跑本脚本即可。
    # data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RAG_DIR.parent / "Huatuo-Lite" / "format_data.jsonl"
    # index_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else RAG_DIR / "vector_index"
    data_path = RAG_DIR.parent / "Huatuo-Lite" / "format_data.jsonl"
    index_dir = RAG_DIR / "vector_index"
    config = MedicalConfig(
        data_path=str(data_path),
        index_save_path=str(index_dir),
    )
    ingestion = IngestionModule(config.data_path)
    ingestion.load_documents()
    print(f"数据版本: {data_path.name} -> 索引目录: {index_dir}", flush=True)

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