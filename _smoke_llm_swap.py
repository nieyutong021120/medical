"""临时冒烟脚本：验证 LLM 提供商切换（Moonshot → 本地 OpenAI 兼容端点）后的关键链路。跑完即删。"""
import sys, json, os, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAG = ROOT / "medical_rag"
sys.path.insert(0, str(RAG))
from dotenv import load_dotenv
load_dotenv(RAG / ".env", override=True)
logging.basicConfig(level=logging.WARNING)

print("[1/5] GET /v1/models 确认模型存在", flush=True)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("API_BASE_URL"))
    models = [m.id for m in client.models.list().data]
    print(f"  端点模型数: {len(models)}", flush=True)
    print(f"  glm-5.3-flash 存在: {'glm-5.3-flash' in models}", flush=True)
    if "glm-5.3-flash" not in models:
        print("  含 glm 的模型:", [m for m in models if "glm" in m.lower()][:10], flush=True)
except Exception as e:
    print(f"  models 列表失败（不阻断后续真实调用测试）: {e}", flush=True)

from config import DEFAULT_CONFIG
from kb_modules.dialogue import DialogueModule
from kb_modules.generation import GenerationModule
from kb_modules.agent import AgentModule
from langchain_core.documents import Document

print(f"[2/5] 意图路由 (model={DEFAULT_CONFIG.llm_model}, temperature={DEFAULT_CONFIG.temperature})", flush=True)
dlg = DialogueModule(model_name=DEFAULT_CONFIG.llm_model, temperature=DEFAULT_CONFIG.temperature)
for q, exp in [("糖尿病有哪些症状？", "medical_qa"),
               ("今天天气如何？", "out_of_scope"),
               ("我最近头晕该怎么办？", "complex_or_ambiguous")]:
    pred = dlg.query_router(q)
    print(f"  {'OK ' if pred == exp else 'MISS'} '{q}' -> {pred} (期望 {exp})", flush=True)

print("[3/5] 查询重写", flush=True)
rw = dlg.query_rewrite("上不来气")
print(f"  '上不来气' -> '{rw}'", flush=True)

print("[4/5] 生成（真实 QA 上下文）", flush=True)
docs = []
with open(ROOT / "Huatuo-Lite" / "format_data.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if "糖尿病" in d["question"] and d.get("score", 0) >= 4:
            docs.append(Document(page_content=d["question"], metadata={
                "answer": d["answer"], "label": d["label"], "score": d["score"],
                "chunk_id": str(d["id"]), "related_diseases": d.get("related_diseases", "")}))
            if len(docs) >= 3:
                break
print(f"  从数据集取到 {len(docs)} 条真实糖尿病 QA", flush=True)
gen = GenerationModule(model_name=DEFAULT_CONFIG.llm_model,
                       temperature=DEFAULT_CONFIG.temperature, max_tokens=DEFAULT_CONFIG.max_tokens)
ans = gen.generate_answer("糖尿病有哪些症状？", docs)
print("  非流式回答(前200字):", ans[:200].replace("\n", " "), flush=True)
n, first = 0, None
for piece in gen.generate_answer_stream("糖尿病有哪些症状？", docs):
    if first is None:
        first = piece
    n += len(piece)
print(f"  流式 OK: 首块={str(first)[:30]!r}, 共 {n} 字符", flush=True)

print("[5/5] Agent 工具调用（stub 检索返回数据集真实 QA，仅验证工具链路）", flush=True)
pool = []
with open(ROOT / "Huatuo-Lite" / "format_data.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20000:
            break
        pool.append(json.loads(line))

class StubRetrieval:
    """冒烟专用：关键词子串匹配真实 QA（非真实混合检索）"""
    def hybrid_search(self, query, top_k=5):
        kws = [w for w in query.replace("？", " ").split() if len(w) >= 2]
        hits = []
        for d in pool:
            text = d["question"] + d["answer"]
            if any(k in text for k in kws):
                hits.append(Document(page_content=d["question"], metadata={
                    "answer": d["answer"], "label": d["label"], "score": d["score"],
                    "chunk_id": str(d["id"])}))
                if len(hits) >= top_k:
                    break
        return hits

agent = AgentModule(StubRetrieval(), model_name=DEFAULT_CONFIG.llm_model,
                    temperature=DEFAULT_CONFIG.temperature, max_iterations=2, top_k=5)
for q in ["我最近头晕该怎么办？", "肚子疼还恶心头晕是怎么回事？"]:
    events = list(agent.run(q))
    tools = [e for e in events if e.get("_event") == "tool"]
    res = [e for e in events if e.get("_event") == "agent_result"][-1]
    print(f"  '{q}' -> tools={[t['name'] for t in tools]} action={res['action']}", flush=True)
    if res["action"] == "clarify":
        print("    澄清问题:", res["clarification"][:80], flush=True)
    if res["action"] == "answer":
        print(f"    检索到 {len(res['chunks'])} 条, rewritten='{res['rewritten']}'", flush=True)

print("冒烟完成", flush=True)