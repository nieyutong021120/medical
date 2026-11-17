"""检索机制离线评测：对比 向量-only / BM25-only / 混合RRF / 混合RRF+score加权(生产配置)。

ground truth 全部来自 Huatuo-Lite 自身元数据，无 LLM 参与、无人工标注：
1. 已知-item 召回率 Recall@1/@5：以库内真实问题为查询，金标=该问题自身 QA。
   三种查询形态模拟真实输入：full=原问题(上限基线)、light=去客套前缀/尾标点(口语)、
   frag=light 截前12字(短片段输入)。
2. 查准率 Precision@5：top-5 中与查询 related_diseases 同病种的比例
   （仅统计查询疾病标签非空的样本；空标签视为不相关）。
3. 导诊准确率：top-5 按各配置自身分数对 label 加权投票 == 查询真实 label。
4. 双路互补性：金标仅被单路 top-10 召回的比例（混合检索的收益来源）。

运行：D:/miniconda/envs/all-in-rag/python.exe -X utf8 medical_rag/scripts/eval_retrieval.py
"""

import os
import re
import sys
import json
import time
import random
import logging
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))
logging.basicConfig(level=logging.WARNING)

from kb_modules.ingestion import IngestionModule
from kb_modules.indexing import IndexingModule
from kb_modules.retrieval import RetrievalModule

N = 300
SEED = 42
RRF_K = 60
TOP_K = 5
RETRIEVER_K = 10

POLITE = re.compile(r"^(医生你好|医生您好|你好|您好|请问|我想问一下|想问一下|我想咨询|想咨询|咨询一下|问一下|我想了解|想了解)[，,、：:\s]*")


def light_query(q: str) -> str:
    prev = None
    while prev != q:
        prev = q
        q = POLITE.sub("", q)
    return q.rstrip("。？?！!，,；;、 \n")


def frag_query(q: str) -> str:
    return light_query(q)[:12]


def rrf_fuse(v_docs, b_docs, weighted: bool):
    scores, objs = {}, {}
    for docs in (v_docs, b_docs):
        for rank, d in enumerate(docs):
            key = d.metadata["chunk_id"]
            objs[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    if weighted:
        scores = {k: v * (1 + 0.05 * (objs[k].metadata.get("score", 4) - 4)) for k, v in scores.items()}
    order = sorted(scores, key=scores.get, reverse=True)
    return [objs[k] for k in order], scores


def vote_label(docs, weights):
    tally = defaultdict(float)
    for d, w in zip(docs[:TOP_K], weights[:TOP_K]):
        tally[d.metadata["label"]] += w
    return max(tally, key=tally.get) if tally else ""


def main():
    ing = IngestionModule(str(RAG_DIR.parent / "Huatuo-Lite" / "format_data.jsonl"))
    ing.load_documents()
    idx = IndexingModule(index_save_path=str(RAG_DIR / "vector_index"))
    vs = idx.load_index()
    bm = idx.load_bm25_index()
    ret = RetrievalModule(vs, bm, retriever_k=RETRIEVER_K)

    rng = random.Random(SEED)
    sample = rng.sample(ing.documents, N)

    variants = {"full": lambda q: q, "light": light_query, "frag": frag_query}
    # recall[variant][config] = [hit@1, hit@5]
    recall = {v: {c: [0, 0] for c in ("V", "B", "H", "HW")} for v in variants}
    precision = {c: [] for c in ("V", "B", "H", "HW")}
    dept_acc = {c: [0, 0] for c in ("V", "B", "H", "HW")}
    complement = {v: {"only_v": 0, "only_b": 0, "both": 0, "neither": 0} for v in variants}
    lat = {"vector_ms": [], "bm25_ms": []}
    n_disease = 0

    for qi, d in enumerate(sample):
        meta = d.metadata
        gold = meta["chunk_id"]
        disease = meta.get("related_diseases", "")
        label = meta["label"]
        for vname, fn in variants.items():
            q = fn(d.page_content)
            t0 = time.perf_counter()
            v_docs = ret.vectorstore.similarity_search(q, k=RETRIEVER_K)
            lat["vector_ms"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            b_docs = ret.bm25_retriever.invoke(q)
            lat["bm25_ms"].append((time.perf_counter() - t0) * 1000)

            v_ids = [x.metadata["chunk_id"] for x in v_docs]
            b_ids = [x.metadata["chunk_id"] for x in b_docs]
            in_v, in_b = gold in v_ids, gold in b_ids
            complement[vname]["both" if in_v and in_b else "only_v" if in_v else "only_b" if in_b else "neither"] += 1

            h_docs, h_scores = rrf_fuse(v_docs, b_docs, weighted=False)
            hw_docs, hw_scores = rrf_fuse(v_docs, b_docs, weighted=True)
            rank_w = [1.0 / (RRF_K + r + 1) for r in range(RETRIEVER_K)]
            configs = {
                "V": (v_docs, rank_w),
                "B": (b_docs, rank_w),
                "H": (h_docs, [h_scores[x.metadata["chunk_id"]] for x in h_docs]),
                "HW": (hw_docs, [hw_scores[x.metadata["chunk_id"]] for x in hw_docs]),
            }
            for c, (docs, weights) in configs.items():
                ids = [x.metadata["chunk_id"] for x in docs[:TOP_K]]
                recall[vname][c][0] += int(gold in ids[:1])
                recall[vname][c][1] += int(gold in ids)
                if vname == "full":
                    if disease:
                        n_disease += 1 if c == "V" else 0
                        precision[c].append(sum(1 for x in docs[:TOP_K] if x.metadata.get("related_diseases", "") == disease) / TOP_K)
                    dept_acc[c][0] += int(vote_label(docs, weights) == label)
                    dept_acc[c][1] += 1
        if (qi + 1) % 50 == 0:
            print(f"progress {qi + 1}/{N}", flush=True)

    out = {
        "n_queries": N,
        "n_disease_queries": n_disease,
        "recall_pct": {
            v: {c: {"at1": round(100 * h[0] / N, 1), "at5": round(100 * h[1] / N, 1)} for c, h in cs.items()}
            for v, cs in recall.items()
        },
        "precision_at5_pct": {c: round(100 * sum(ps) / len(ps), 1) for c, ps in precision.items()},
        "dept_acc_pct": {c: round(100 * a / b, 1) for c, (a, b) in dept_acc.items()},
        "complement_top10": complement,
        "latency_ms_mean": {k: round(sum(v) / len(v), 1) for k, v in lat.items()},
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()