"""检索评测 v2：在 v1(eval_retrieval.py) 基础上升级——

1. 新增 MRR@5（金标在 top-5 内的排名倒数均值，衡量排序质量）
2. 新增难查询形态：kw=jieba 关键词抽取（模拟 Agent 改写后的检索词）、
   para=LLM 口语改写（--build-para 离线生成缓存，评测时零 API 调用）
3. 新增配置：HB2=BM25 路权重×2 的加权 RRF（验证短查询下 BM25 主导假设）、
   HR=混合召回 top-20 + bge-reranker-base 交叉编码器精排（--rerank 启用）
4. --score-coef 可调质量分加权系数（默认 0.05，供扫参）

v1 的 V/B/H/HW × full/light/frag 口径保持不变，结果可与 eval_results.json 对比。

运行（评测，纯离线）：
  python -X utf8 medical_rag/scripts/eval_retrieval_v2.py
  python -X utf8 medical_rag/scripts/eval_retrieval_v2.py --rerank
运行（生成口语改写缓存，耗 LLM 调用，RPM=3 → 每条间隔 21s，支持断点续跑）：
  python -X utf8 medical_rag/scripts/eval_retrieval_v2.py --build-para 300
"""

import os
import re
import sys
import json
import time
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
SCRIPTS_DIR = Path(__file__).resolve().parent
RAG_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(RAG_DIR))
logging.basicConfig(level=logging.WARNING)

RRF_K = 60
TOP_K = 5
RETRIEVER_K = 10
RERANK_CANDIDATES = 20

POLITE = re.compile(r"^(医生你好|医生您好|你好|您好|请问|我想问一下|想问一下|我想咨询|想咨询|咨询一下|问一下|我想了解|想了解)[，,、：:\s]*")


def light_query(q: str) -> str:
    prev = None
    while prev != q:
        prev = q
        q = POLITE.sub("", q)
    return q.rstrip("。？?！!，,；;、 \n")


def frag_query(q: str) -> str:
    return light_query(q)[:12]


def kw_query(q: str) -> str:
    """jieba 词性过滤抽取实词（名/动/形），模拟关键词形态检索词；失败回退 frag"""
    try:
        import jieba.posseg as pseg
        words, seen = [], set()
        for w, flag in pseg.cut(light_query(q)):
            if flag.startswith(("n", "v", "a")) and w not in seen and len(w) > 1:
                seen.add(w)
                words.append(w)
            if len(words) >= 4:
                break
        return " ".join(words) if words else frag_query(q)
    except Exception:
        return frag_query(q)


def rrf_fuse(v_docs, b_docs, score_coef: float = 0.0, w_v: float = 1.0, w_b: float = 1.0):
    """加权 RRF：w_v/w_b 为两路权重；score_coef>0 时叠加 QA 质量分加权"""
    scores, objs = {}, {}
    for docs, w in ((v_docs, w_v), (b_docs, w_b)):
        for rank, d in enumerate(docs):
            key = d.metadata["chunk_id"]
            objs[key] = d
            scores[key] = scores.get(key, 0.0) + w / (RRF_K + rank + 1)
    if score_coef:
        scores = {k: v * (1 + score_coef * (objs[k].metadata.get("score", 4) - 4)) for k, v in scores.items()}
    order = sorted(scores, key=scores.get, reverse=True)
    return [objs[k] for k in order], scores


def mrr_at_k(docs, gold: str, k: int = TOP_K) -> float:
    for rank, d in enumerate(docs[:k], 1):
        if d.metadata["chunk_id"] == gold:
            return 1.0 / rank
    return 0.0


def vote_label(docs, weights):
    tally = defaultdict(float)
    for d, w in zip(docs[:TOP_K], weights[:TOP_K]):
        tally[d.metadata["label"]] += w
    return max(tally, key=tally.get) if tally else ""


def build_para(args):
    """离线生成口语改写缓存：chunk_id → para，逐条落盘可续跑"""
    from dotenv import load_dotenv
    load_dotenv(RAG_DIR / ".env")
    load_dotenv(RAG_DIR.parent / ".env")
    from langchain_openai import ChatOpenAI
    from kb_modules.ingestion import IngestionModule

    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError("请设置 MOONSHOT_API_KEY")
    llm = ChatOpenAI(model="kimi-k3", base_url="https://api.moonshot.cn/v1",
                     api_key=api_key, temperature=1, max_tokens=256, max_retries=5)
    ing = IngestionModule(str(RAG_DIR.parent / "Huatuo-Lite" / "format_data.jsonl"))
    ing.load_documents()
    sample = random.Random(args.seed).sample(ing.documents, args.build_para)

    cache_path = Path(args.para_cache)
    done = set()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["chunk_id"])
    todo = [d for d in sample if d.metadata["chunk_id"] not in done]
    print(f"口语改写：目标 {len(sample)}，已完成 {len(done)}，待生成 {len(todo)}")
    with cache_path.open("a", encoding="utf-8") as f:
        for i, d in enumerate(todo):
            prompt = ("把下面的医疗咨询问题改写为患者的口语化表述（可含错别字或方言色彩），"
                      "保持医学含义不变，不添加新症状，只输出改写后的一句话：\n" + d.page_content)
            try:
                para = llm.invoke(prompt).content.strip().split("\n")[0]
            except Exception as e:
                logging.warning(f"改写失败 {d.metadata['chunk_id']}: {e}")
                continue
            f.write(json.dumps({"chunk_id": d.metadata["chunk_id"], "para": para}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(todo)}] {d.page_content[:30]} -> {para[:30]}", flush=True)
            if i < len(todo) - 1:
                time.sleep(args.interval)
    print(f"缓存已保存: {cache_path}")


def main():
    ap = argparse.ArgumentParser(description="检索评测 v2")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rerank", action="store_true", help="启用 bge-reranker-base 精排配置 HR")
    ap.add_argument("--rerank-k", type=int, default=RERANK_CANDIDATES,
                    help="精排候选条数（默认 20；设 10 可与 k=20 结果对比候选池贡献）")
    ap.add_argument("--score-coef", type=float, default=0.05, help="HW 配置的质量分加权系数")
    ap.add_argument("--para-cache", default=str(SCRIPTS_DIR / "eval_para_queries.jsonl"))
    ap.add_argument("--build-para", type=int, default=0, help="生成 N 条口语改写缓存后退出")
    ap.add_argument("--interval", type=float, default=21.0)
    ap.add_argument("--out", default=str(SCRIPTS_DIR / "eval_retrieval_v2_results.json"))
    args = ap.parse_args()

    if args.build_para:
        build_para(args)
        return

    from kb_modules.ingestion import IngestionModule
    from kb_modules.indexing import IndexingModule

    ing = IngestionModule(str(RAG_DIR.parent / "Huatuo-Lite" / "format_data.jsonl"))
    ing.load_documents()
    idx = IndexingModule(index_save_path=str(RAG_DIR / "vector_index"))
    vs = idx.load_index()
    bm = idx.load_bm25_index()
    fetch_k = args.rerank_k if args.rerank else RETRIEVER_K
    bm.k = fetch_k

    reranker = None
    if args.rerank:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder("BAAI/bge-reranker-base")
            print("Reranker 加载成功: bge-reranker-base")
        except Exception as e:
            print(f"Reranker 加载失败（跳过 HR 配置）: {e}\n"
                  f"如需下载模型，去掉 HF_HUB_OFFLINE 环境变量后重试")

    para_map = {}
    if Path(args.para_cache).exists():
        for line in Path(args.para_cache).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            para_map[r["chunk_id"]] = r["para"]
        print(f"口语改写缓存: {len(para_map)} 条")

    sample = random.Random(args.seed).sample(ing.documents, args.n)
    variants = ["full", "light", "frag", "kw"] + (["para"] if para_map else [])
    configs = ["V", "B", "H", "HW", "HB2"] + (["HR"] if reranker else [])

    # stats[variant][config] = 累计指标
    stats = {v: {c: {"hit1": 0, "hit5": 0, "mrr": 0.0, "n": 0, "prec": [], "dept": [0, 0]}
                 for c in configs} for v in variants}
    lat = {"vector_ms": [], "bm25_ms": [], "rerank_ms": []}
    complement = {v: {"only_v": 0, "only_b": 0, "both": 0, "neither": 0} for v in variants}

    for qi, d in enumerate(sample):
        meta = d.metadata
        gold = meta["chunk_id"]
        disease = meta.get("related_diseases", "")
        label = meta["label"]
        queries = {"full": d.page_content, "light": light_query(d.page_content),
                   "frag": frag_query(d.page_content), "kw": kw_query(d.page_content)}
        if para_map:
            queries["para"] = para_map.get(gold)

        for vname, q in queries.items():
            if not q:
                continue
            t0 = time.perf_counter()
            v_docs = vs.similarity_search(q, k=fetch_k)
            lat["vector_ms"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            b_docs = bm.invoke(q)
            lat["bm25_ms"].append((time.perf_counter() - t0) * 1000)

            v10, b10 = v_docs[:RETRIEVER_K], b_docs[:RETRIEVER_K]
            v_ids = [x.metadata["chunk_id"] for x in v10]
            b_ids = [x.metadata["chunk_id"] for x in b10]
            in_v, in_b = gold in v_ids, gold in b_ids
            complement[vname]["both" if in_v and in_b else "only_v" if in_v else "only_b" if in_b else "neither"] += 1

            rank_w = [1.0 / (RRF_K + r + 1) for r in range(fetch_k)]
            h_docs, h_scores = rrf_fuse(v10, b10)
            hw_docs, hw_scores = rrf_fuse(v10, b10, score_coef=args.score_coef)
            hb2_docs, hb2_scores = rrf_fuse(v10, b10, w_b=2.0)
            cfg_docs = {
                "V": (v10, rank_w), "B": (b10, rank_w),
                "H": (h_docs, [h_scores[x.metadata["chunk_id"]] for x in h_docs]),
                "HW": (hw_docs, [hw_scores[x.metadata["chunk_id"]] for x in hw_docs]),
                "HB2": (hb2_docs, [hb2_scores[x.metadata["chunk_id"]] for x in hb2_docs]),
            }
            if reranker:
                cand, _ = rrf_fuse(v_docs, b_docs)
                cand = cand[:args.rerank_k]
                pairs = [(q, (x.page_content + "\n" + x.metadata.get("answer", ""))[:512]) for x in cand]
                t0 = time.perf_counter()
                rr_scores = reranker.predict(pairs)
                lat["rerank_ms"].append((time.perf_counter() - t0) * 1000)
                order = sorted(range(len(cand)), key=lambda i: rr_scores[i], reverse=True)
                hr_docs = [cand[i] for i in order]
                cfg_docs["HR"] = (hr_docs, [float(rr_scores[i]) for i in order])

            for c, (docs, weights) in cfg_docs.items():
                s = stats[vname][c]
                top5 = docs[:TOP_K]
                ids = [x.metadata["chunk_id"] for x in top5]
                s["hit1"] += int(gold in ids[:1])
                s["hit5"] += int(gold in ids)
                s["mrr"] += mrr_at_k(docs, gold)
                s["n"] += 1
                if vname == "full":
                    if disease:
                        s["prec"].append(sum(1 for x in top5 if x.metadata.get("related_diseases", "") == disease) / TOP_K)
                    s["dept"][0] += int(vote_label(docs, weights) == label)
                    s["dept"][1] += 1
        if (qi + 1) % 50 == 0:
            print(f"progress {qi + 1}/{len(sample)}", flush=True)

    out = {"n_queries": args.n, "score_coef": args.score_coef, "rerank_enabled": bool(reranker),
           "recall_mrr_pct": {}, "precision_at5_pct": {}, "dept_acc_pct": {},
           "complement_top10": complement,
           "latency_ms_mean": {k: round(sum(v) / len(v), 1) for k, v in lat.items() if v}}
    for v in variants:
        out["recall_mrr_pct"][v] = {}
        for c in configs:
            s = stats[v][c]
            if not s["n"]:
                continue
            out["recall_mrr_pct"][v][c] = {
                "n": s["n"],
                "recall_at1": round(100 * s["hit1"] / s["n"], 1),
                "recall_at5": round(100 * s["hit5"] / s["n"], 1),
                "mrr_at5": round(100 * s["mrr"] / s["n"], 1),
            }
            if s["prec"]:
                out["precision_at5_pct"][c] = round(100 * sum(s["prec"]) / len(s["prec"]), 1)
            if s["dept"][1]:
                out["dept_acc_pct"][c] = round(100 * s["dept"][0] / s["dept"][1], 1)

    print(json.dumps(out, ensure_ascii=False, indent=1))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已保存: {args.out}")

    print(f"\n{'变体':<8}{'配置':<6}{'R@1':>7}{'R@5':>7}{'MRR@5':>8}")
    for v in variants:
        for c in configs:
            r = out["recall_mrr_pct"][v].get(c)
            if r:
                print(f"{v:<8}{c:<6}{r['recall_at1']:>6}%{r['recall_at5']:>6}%{r['mrr_at5']:>7}%")


if __name__ == "__main__":
    main()