"""意图路由独立评测：只调 query_router，不跑检索/生成，隔离变量。

- 测试集：eval_routing_cases.jsonl（60 条人工标注，三类各 20，含边界/多轮历史/追问回答/伪装越界难例）
- 指标：总准确率、每类 precision/recall/F1、3x3 混淆矩阵、按难例标签分组准确率
- 限速：Moonshot RPM=3，默认每条间隔 21s（60 条约 21 分钟）
- 断点续跑：逐条写入 *_progress.jsonl，--resume 跳过已完成 id
- 数据集体检：--check 只校验测试集格式与配比，不调 LLM、不需要 API key

运行：
  D:/miniconda/envs/all-in-rag/python.exe -X utf8 medical_rag/scripts/eval_routing.py --check
  D:/miniconda/envs/all-in-rag/python.exe -X utf8 medical_rag/scripts/eval_routing.py --limit 6
  D:/miniconda/envs/all-in-rag/python.exe -X utf8 medical_rag/scripts/eval_routing.py --resume
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict

SCRIPTS_DIR = Path(__file__).resolve().parent
RAG_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(RAG_DIR))
logging.basicConfig(level=logging.WARNING)

LABELS = ["medical_qa", "complex_or_ambiguous", "out_of_scope"]


def load_cases(path: Path):
    cases = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        c = json.loads(line)
        assert c["expected"] in LABELS, f"line {ln}: 非法标签 {c['expected']}"
        assert isinstance(c.get("history", []), list), f"line {ln}: history 必须是列表"
        for m in c["history"]:
            assert m["role"] in ("user", "assistant") and "text" in m, f"line {ln}: history 消息格式错误"
        cases.append(c)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "存在重复 id"
    return cases


def check_dataset(cases):
    print(f"用例总数: {len(cases)}")
    print("类别分布:", dict(Counter(c["expected"] for c in cases)))
    print("难例分布:", dict(Counter(c["tag"] for c in cases)))
    n_hist = sum(1 for c in cases if c["history"])
    print(f"带对话历史用例: {n_hist}")
    for label in LABELS:
        tagged = Counter(c["tag"] for c in cases if c["expected"] == label)
        print(f"  {label}: {dict(tagged)}")


def compute_metrics(records):
    valid = [r for r in records if r["pred"] in LABELS]
    n_error = len(records) - len(valid)
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for r in valid:
        confusion[r["expected"]][r["pred"]] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[t][label] for t in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "precision_pct": round(100 * precision, 2),
            "recall_pct": round(100 * recall, 2),
            "f1_pct": round(100 * f1, 2),
        }
    by_tag = defaultdict(lambda: [0, 0])
    for r in valid:
        by_tag[r["tag"]][0] += int(r["pred"] == r["expected"])
        by_tag[r["tag"]][1] += 1
    correct = sum(confusion[l][l] for l in LABELS)
    return {
        "n_cases": len(records),
        "n_api_errors": n_error,
        "accuracy_pct": round(100 * correct / len(valid), 2) if valid else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "by_tag_pct": {t: round(100 * a / b, 2) for t, (a, b) in by_tag.items()},
        "errors": [r for r in valid if r["pred"] != r["expected"]],
    }


def print_report(m):
    print("\n" + "=" * 60)
    print(f"总准确率: {m['accuracy_pct']}%  (有效 {m['n_cases'] - m['n_api_errors']}/{m['n_cases']}，API失败 {m['n_api_errors']})")
    print(f"\n{'类别':<24}{'样本':>4}{'精确率':>9}{'召回率':>9}{'F1':>8}")
    for label, s in m["per_class"].items():
        print(f"{label:<24}{s['support']:>4}{s['precision_pct']:>8}%{s['recall_pct']:>8}%{s['f1_pct']:>7}%")
    print("\n混淆矩阵 (行=真实, 列=预测):")
    header = f"{'':<24}" + "".join(f"{l[:12]:>14}" for l in LABELS)
    print(header)
    for t in LABELS:
        print(f"{t:<24}" + "".join(f"{m['confusion_matrix'][t][p]:>14}" for p in LABELS))
    print("\n按难例标签准确率:", json.dumps(m["by_tag_pct"], ensure_ascii=False))
    if m["errors"]:
        print(f"\n错例 {len(m['errors'])} 条:")
        for r in m["errors"]:
            print(f"  id={r['id']} [{r['tag']}] 真实={r['expected']} 预测={r['pred']} | {r['question'][:40]}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="意图路由独立评测")
    ap.add_argument("--cases", default=str(SCRIPTS_DIR / "eval_routing_cases.jsonl"))
    ap.add_argument("--out", default=str(SCRIPTS_DIR / "eval_routing_results.json"))
    ap.add_argument("--interval", type=float, default=21.0, help="每次调用间隔秒数（RPM=3 → ≥20s）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟测试）")
    ap.add_argument("--resume", action="store_true", help="跳过 progress 文件中已完成的 id")
    ap.add_argument("--check", action="store_true", help="只校验测试集，不调 LLM")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    if args.check:
        check_dataset(cases)
        return
    if args.limit:
        cases = cases[:args.limit]

    progress_path = Path(args.out).with_name(Path(args.out).stem + "_progress.jsonl")
    records = {}
    if args.resume and progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            records[r["id"]] = r
        print(f"续跑：已完成 {len(records)} 条")

    todo = [c for c in cases if c["id"] not in records]
    if todo:
        from dotenv import load_dotenv
        load_dotenv(RAG_DIR / ".env")
        load_dotenv(RAG_DIR.parent / ".env")
        from config import DEFAULT_CONFIG
        from kb_modules.dialogue import DialogueModule
        dialogue = DialogueModule(model_name=DEFAULT_CONFIG.llm_model)

    with progress_path.open("a", encoding="utf-8") as pf:
        for i, c in enumerate(todo):
            t0 = time.perf_counter()
            try:
                pred = dialogue.query_router(c["question"], c["history"] or None)
            except Exception as e:
                logging.warning(f"id={c['id']} 调用失败: {e}")
                pred = "_error"
            r = {**c, "pred": pred, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            records[c["id"]] = r
            pf.write(json.dumps(r, ensure_ascii=False) + "\n")
            pf.flush()
            mark = "OK " if pred == c["expected"] else "MISS"
            print(f"[{i + 1}/{len(todo)}] {mark} id={c['id']} 真实={c['expected']} 预测={pred}", flush=True)
            if i < len(todo) - 1:
                time.sleep(args.interval)

    ordered = [records[c["id"]] for c in cases if c["id"] in records]
    m = compute_metrics(ordered)
    print_report(m)
    out = {"summary": {k: v for k, v in m.items() if k != "errors"},
           "errors": m["errors"], "records": ordered}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()