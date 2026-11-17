"""
数据摄入模块 - 华佗QA对加载（改自 hr_compliance_rag/kb_modules/ingestion.py）

旧：法规Markdown按条分块(父文档/子块两级)。QA对天然即一个检索单元，无需分块。
关键设计：向量侧 page_content=question（question-to-question 匹配提升召回精度），
BM25侧语料=question+answer（药名/症状词仅出现在答案中时也能被关键词召回）。
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IngestionModule:
    """数据摄入模块 - QA对加载与统计"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []
        self._stats = None

    def load_documents(self) -> List[Document]:
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        logger.info(f"正在加载QA数据: {path}")
        docs, skipped = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                q = (d.get("question") or "").strip()
                a = (d.get("answer") or "").strip()
                if not q or not a:
                    skipped += 1
                    continue
                docs.append(Document(
                    page_content=q,
                    metadata={
                        "chunk_id": str(d["id"]),
                        "answer": a,
                        "score": int(d.get("score", 4)),
                        "label": d.get("label", "其他"),
                        "related_diseases": d.get("related_diseases", ""),
                        "source": "Huatuo-Lite",
                    },
                ))
        self.documents = docs
        logger.info(f"加载完成: {len(docs)} 条QA对，跳过空字段 {skipped} 条")
        return docs

    def get_bm25_documents(self) -> List[Document]:
        """BM25语料：question+answer，chunk_id 与向量侧一致供RRF去重"""
        return [
            Document(
                page_content=doc.page_content + "\n" + doc.metadata["answer"],
                metadata=doc.metadata,
            )
            for doc in self.documents
        ]

    def get_statistics(self) -> dict:
        if self._stats is None:
            labels = Counter(d.metadata["label"] for d in self.documents)
            scores = Counter(d.metadata["score"] for d in self.documents)
            avg_ans = (sum(len(d.metadata["answer"]) for d in self.documents) / len(self.documents)) if self.documents else 0
            self._stats = {
                "total_qa": len(self.documents),
                "departments": dict(labels.most_common()),
                "scores": {str(k): v for k, v in sorted(scores.items())},
                "avg_answer_len": round(avg_ans),
                "dataset": "Huatuo-Lite (FreedomIntelligence/Huatuo26M-Lite)",
            }
        return self._stats

    @staticmethod
    def get_supported_departments() -> List[str]:
        return [
            "妇产科", "内科", "皮肤性病科", "儿科", "眼耳鼻喉科", "肿瘤科",
            "神经科学", "外科", "男性健康科", "感染与免疫科", "口腔科",
            "心理科学", "中医科", "生殖健康科", "急诊科", "其他",
        ]