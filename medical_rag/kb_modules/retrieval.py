"""
检索优化模块 - 混合检索(向量 + jieba-BM25) + RRF重排 + 质量评分加权 + 科室过滤
（改自 hr_compliance_rag/kb_modules/retrieval.py）

旧：RRF叠加"效力级别"加权(法律>行政法规>...)。现改为QA质量评分加权：
final = rrf * (1 + 0.05 * (score - 4))，score=5 的参考问答适度优先。
BM25索引改由 IndexingModule 构建/缓存后注入（语料为question+answer）。
"""

import logging
import hashlib
from typing import List, Dict, Any, Callable, Optional

from langchain_community.vectorstores import FAISS
# 旧：from langchain_community.retrievers import BM25Retriever（类型提示放宽以兼容倒排实现）
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalModule:
    """检索优化模块 - 混合检索与过滤"""

    # 旧：def __init__(self, vectorstore: FAISS, bm25_retriever: BM25Retriever, retriever_k: int = 10):
    def __init__(self, vectorstore: FAISS, bm25_retriever: BaseRetriever, retriever_k: int = 10):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.retriever_k = retriever_k
        self.bm25_retriever.k = retriever_k
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.retriever_k}
        )
        logger.info("检索器设置完成")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _match_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in meta:
                return False
            if isinstance(value, list):
                if meta[key] not in value:
                    return False
            elif meta[key] != value:
                return False
        return True

    def _vector_search(self, query: str, k: int, filters: Dict[str, Any] = None) -> List[Document]:
        if filters:
            predicate: Callable[[Dict[str, Any]], bool] = lambda meta: self._match_filters(meta, filters)
            return self.vectorstore.similarity_search(query, k=k, filter=predicate)
        return self.vector_retriever.invoke(query)

    # ------------------------------------------------------------------ #
    def hybrid_search(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Document]:
        """混合检索：向量(可前置过滤) + BM25(后置过滤兜底) -> RRF融合 -> 评分加权 -> top_k"""
        vector_docs = self._vector_search(query, self.retriever_k, filters)

        bm25_docs = self.bm25_retriever.invoke(query)
        if filters:
            bm25_docs = [d for d in bm25_docs if self._match_filters(d.metadata, filters)]

        reranked = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked[:top_k]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        return self.hybrid_search(query, top_k=top_k, filters=filters)

    # ------------------------------------------------------------------ #
    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """RRF重排 + 质量评分加权。去重键用 chunk_id（QA唯一id）"""
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}

        def _key(doc: Document) -> str:
            cid = doc.metadata.get("chunk_id")
            if cid:
                return f"id::{cid}"
            return "md5::" + hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()

        def _accumulate(docs: List[Document], tag: str):
            for rank, doc in enumerate(docs):
                key = _key(doc)
                doc_objects[key] = doc
                rrf = 1.0 / (k + rank + 1)
                doc_scores[key] = doc_scores.get(key, 0.0) + rrf

        _accumulate(vector_docs, "向量检索")
        _accumulate(bm25_docs, "BM25检索")

        # 旧：law_type 效力级别加权；现：score 质量评分加权
        scored = []
        for key, rrf in doc_scores.items():
            score = doc_objects[key].metadata.get("score", 4)
            final = rrf * (1 + 0.05 * (score - 4))
            doc_objects[key].metadata['rrf_score'] = final
            scored.append((final, key))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [doc_objects[key] for _, key in scored]

        logger.info(
            f"RRF重排完成: 向量{len(vector_docs)} + BM25{len(bm25_docs)} -> 合并{len(reranked)}条QA"
        )
        return reranked