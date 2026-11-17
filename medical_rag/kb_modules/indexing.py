"""
索引构建模块 - FAISS向量索引 + jieba-BM25索引的构建/持久化
（改自 hr_compliance_rag/kb_modules/indexing.py，新增BM25索引落盘缓存，
避免17.8万条语料每次启动都重新jieba分词）

方案A提速：新增 InvertedBM25Retriever 倒排索引（term→postings，numpy向量化打分），
分数与 rank_bm25.BM25Okapi 逐位一致；load_bm25_index 优先加载 inverted.pkl。
"""

import logging
import pickle
from typing import Any, List, Optional
from pathlib import Path

import jieba
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

logger = logging.getLogger(__name__)


def _jieba_tokenize(text: str) -> List[str]:
    """中文分词(BM25 预处理)。模块级函数以便pickle按引用序列化。"""
    return [w for w in jieba.cut(text) if w.strip()]


class InvertedBM25Retriever(BaseRetriever):
    """倒排索引BM25检索器（方案A）：分数与 rank_bm25.BM25Okapi 逐位一致。

    postings: term -> (doc_ids int32, tfs int32)，仅含该词的文档；
    idf/doc_len/avgdl/k1/b 原样复用原 BM25Okapi 统计量（含负idf的epsilon下限）。
    查询成本从 O(N·|q|) 的Python全库扫描降为 O(Σ|postings(q_t)|) 的numpy向量运算。
    逐位一致的关键：运算分组、查询词顺序（不去重）、argsort 均与原实现相同；
    原式对 tf=0 文档贡献恰为0，倒排跳过它们不改变任何分数。
    """

    postings: Any = None
    idf: Any = None
    doc_len: Any = None
    norm: Any = None
    avgdl: float = 0.0
    k1: float = 1.5
    b: float = 0.75
    docs: List[Document] = Field(repr=False)
    k: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(len(self.docs))
        for t in query_tokens:
            post = self.postings.get(t)
            if post is None:
                continue
            ids, tfs = post
            # 分组与 rank_bm25 一致: idf * (tf*(k1+1) / (tf + k1*(1-b+b*len/avgdl)))
            scores[ids] += (self.idf.get(t) or 0) * (
                tfs * (self.k1 + 1) / (tfs + self.norm[ids])
            )
        return scores

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        scores = self.get_scores(_jieba_tokenize(query))
        top_n = np.argsort(scores)[::-1][: self.k]
        return [self.docs[i] for i in top_n]


def convert_bm25_to_inverted(old: BM25Retriever, progress_every: int = 20000) -> InvertedBM25Retriever:
    """从现有 BM25Retriever(bm25.pkl) 离线转换：复用分词与统计量，不重新jieba。

    仅把 doc_freqs(文档→{词:tf}) 重排为 postings(词→(ids,tfs))，
    norm 用原表达式预计算，保证查询时逐元素浮点结果与原实现相同。
    """
    v = old.vectorizer
    buckets = {}
    n = len(v.doc_freqs)
    for did, freq in enumerate(v.doc_freqs):
        for term, tf in freq.items():
            if term in buckets:
                buckets[term].append((did, tf))
            else:
                buckets[term] = [(did, tf)]
        if progress_every and (did + 1) % progress_every == 0:
            print(f"convert progress {did + 1}/{n}", flush=True)
    postings = {
        t: (np.array([p[0] for p in lst], dtype=np.int32),
            np.array([p[1] for p in lst], dtype=np.int32))
        for t, lst in buckets.items()
    }
    doc_len = np.array(v.doc_len)
    norm = v.k1 * (1 - v.b + v.b * doc_len / v.avgdl)
    return InvertedBM25Retriever(
        postings=postings, idf=v.idf, doc_len=doc_len, norm=norm,
        avgdl=v.avgdl, k1=v.k1, b=v.b, docs=old.docs, k=old.k,
    )


class IndexingModule:
    """索引构建模块 - 向量化与双索引持久化"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "./vector_index"):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.bm25_retriever = None
        self.setup_embeddings()

    def setup_embeddings(self):
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("嵌入模型初始化完成")

    # ------------------------------------------------------------------ #
    # FAISS 向量索引
    # ------------------------------------------------------------------ #
    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        logger.info(f"正在构建FAISS向量索引({len(chunks)}条，CPU编码耗时较长)...")
        if not chunks:
            raise ValueError("文档块列表不能为空")
        self.vectorstore = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self.vectorstore

    def save_index(self):
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到: {self.index_save_path}")
        if self.bm25_retriever is not None:
            with open(Path(self.index_save_path) / "bm25.pkl", "wb") as f:
                pickle.dump(self.bm25_retriever, f)
            logger.info("BM25索引已缓存到 bm25.pkl")
            # 重建时同步产出倒排索引，加载路径直接拿到快版本（转换秒级，相对嵌入构建可忽略）
            if isinstance(self.bm25_retriever, BM25Retriever):
                inverted = convert_bm25_to_inverted(self.bm25_retriever)
                with open(Path(self.index_save_path) / "inverted.pkl", "wb") as f:
                    pickle.dump(inverted, f)
                logger.info("倒排BM25索引已缓存到 inverted.pkl")

    def load_index(self) -> Optional[FAISS]:
        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None
        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None

    # ------------------------------------------------------------------ #
    # BM25 关键词索引
    # ------------------------------------------------------------------ #
    def build_bm25_index(self, bm25_docs: List[Document], k: int = 10) -> BM25Retriever:
        logger.info(f"正在构建jieba-BM25索引({len(bm25_docs)}条)...")
        self.bm25_retriever = BM25Retriever.from_documents(
            bm25_docs, k=k, preprocess_func=_jieba_tokenize,
        )
        logger.info("BM25索引构建完成")
        return self.bm25_retriever

    # 旧：只加载 bm25.pkl（rank_bm25 前向结构，查询~2.2s）；现优先 inverted.pkl（方案A）
    # def load_bm25_index(self) -> Optional[BM25Retriever]:
    def load_bm25_index(self) -> Optional[BaseRetriever]:
        inv = Path(self.index_save_path) / "inverted.pkl"
        pkl = Path(self.index_save_path) / "bm25.pkl"
        # 过期保护：bm25.pkl 若重建过（比 inverted 新），回退旧索引，避免用到过期 postings
        if inv.exists() and (not pkl.exists() or inv.stat().st_mtime >= pkl.stat().st_mtime):
            try:
                with open(inv, "rb") as f:
                    self.bm25_retriever = pickle.load(f)
                logger.info("倒排BM25索引已从 inverted.pkl 加载")
                return self.bm25_retriever
            except Exception as e:
                logger.warning(f"加载 inverted.pkl 失败: {e}，回退 bm25.pkl")
        if not pkl.exists():
            return None
        try:
            with open(pkl, "rb") as f:
                self.bm25_retriever = pickle.load(f)
            logger.info("BM25索引已从缓存加载（前向；运行 scripts/build_inverted_bm25.py 可提速）")
            return self.bm25_retriever
        except Exception as e:
            logger.warning(f"加载BM25缓存失败: {e}，将重建")
            return None