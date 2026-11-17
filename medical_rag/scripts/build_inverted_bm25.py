"""方案A：把现有 bm25.pkl 离线转换为倒排索引 inverted.pkl。

复用原 jieba 分词与 idf/doc_len/avgdl/k1/b 统计量（不重新分词），
仅把 文档→{词:tf} 重排为 词→(doc_ids,tfs)。落盘前强制通过：
1. 分数逐位一致：np.array_equal(旧get_scores, 新get_scores)
2. top-10 文档下标一致 + invoke 返回的 chunk_id 序列一致
3. 边界查询：重复查询词（不去重）、语料库外词（OOV）
4. pickle 往返后复验
并打印新旧延迟对比。任一断言失败则不写 inverted.pkl。

运行：D:/miniconda/envs/all-in-rag/python.exe -X utf8 medical_rag/scripts/build_inverted_bm25.py
"""

import os
import sys
import time
import pickle
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))

from kb_modules.indexing import (
    InvertedBM25Retriever,
    convert_bm25_to_inverted,
    _jieba_tokenize,
)

IDX = RAG_DIR / "vector_index"
N_CHECK = 20
SEED = 42


def main():
    t0 = time.perf_counter()
    with open(IDX / "bm25.pkl", "rb") as f:
        old = pickle.load(f)
    print(f"bm25.pkl 加载: {len(old.docs)} docs, {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    new = convert_bm25_to_inverted(old)
    n_post = sum(len(ids) for ids, _ in new.postings.values())
    print(f"倒排转换完成: {len(new.postings)} terms / {n_post} postings, "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    rng = random.Random(SEED)
    sample = rng.sample(old.docs, N_CHECK)
    queries = [d.page_content[:40] for d in sample]
    queries += ["发烧 发烧 怎么办", "语料库外词汇测试xyzabc感冒"]

    old_lat, new_lat = [], []
    for i, q in enumerate(queries):
        tokens = _jieba_tokenize(q)
        t0 = time.perf_counter()
        s_old = old.vectorizer.get_scores(tokens)
        old_lat.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        s_new = new.get_scores(tokens)
        new_lat.append((time.perf_counter() - t0) * 1000)

        assert np.array_equal(s_old, s_new), f"分数不一致 query {i}: {q!r}"
        assert np.array_equal(np.argsort(s_old)[::-1][:10],
                              np.argsort(s_new)[::-1][:10]), f"top10 不一致 query {i}"
        docs_old = old.invoke(q)
        docs_new = new.invoke(q)
        assert [d.metadata["chunk_id"] for d in docs_old] == \
               [d.metadata["chunk_id"] for d in docs_new], f"invoke 不一致 query {i}"
        print(f"parity {i + 1}/{len(queries)} OK  old={old_lat[-1]:.0f}ms "
              f"new={new_lat[-1]:.1f}ms", flush=True)

    speedup = float(np.mean(old_lat) / np.mean(new_lat))
    print(f"ALL PARITY PASSED ({len(queries)} queries). "
          f"mean old={np.mean(old_lat):.0f}ms new={np.mean(new_lat):.2f}ms "
          f"speedup={speedup:.0f}x", flush=True)

    t0 = time.perf_counter()
    with open(IDX / "inverted.pkl", "wb") as f:
        pickle.dump(new, f)
    size_mb = (IDX / "inverted.pkl").stat().st_size / 1e6
    print(f"inverted.pkl 已保存 ({size_mb:.0f}MB, {time.perf_counter() - t0:.1f}s)", flush=True)

    with open(IDX / "inverted.pkl", "rb") as f:
        reloaded = pickle.load(f)
    assert isinstance(reloaded, InvertedBM25Retriever)
    q = queries[0]
    assert np.array_equal(reloaded.get_scores(_jieba_tokenize(q)),
                          new.get_scores(_jieba_tokenize(q))), "pickle 往返后分数不一致"
    assert [d.metadata["chunk_id"] for d in reloaded.invoke(q)] == \
           [d.metadata["chunk_id"] for d in new.invoke(q)]
    print("pickle 往返复验 OK", flush=True)


if __name__ == "__main__":
    main()