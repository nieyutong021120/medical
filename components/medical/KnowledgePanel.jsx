"use client";

// 知识库概览（改自 components/hr/KnowledgePanel.jsx）：
// 旧：浏览24条虚构政策。现：数据集统计 + 真实样例QA浏览（完整17.8万条在后端索引中）
import { useEffect, useMemo, useRef, useState } from "react";
import { animate, stagger } from "animejs";
import Icon from "./Icons.jsx";
import { SAMPLE_QA, DEPARTMENTS, FALLBACK_STATS } from "../../data/medicalKnowledge.js";

export default function KnowledgePanel({ onOpenEntry, onAsk, stats }) {
  const [query, setQuery] = useState("");
  const [dept, setDept] = useState("all");
  const gridRef = useRef(null);
  const s = stats || FALLBACK_STATS;

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return SAMPLE_QA.filter((e) => {
      if (dept !== "all" && e.label !== dept) return false;
      if (!q) return true;
      return (e.question + e.answer + e.related_diseases).toLowerCase().includes(q);
    });
  }, [query, dept]);

  useEffect(() => {
    if (!gridRef.current) return;
    const cards = gridRef.current.querySelectorAll(".kb-card");
    if (cards.length) {
      animate(cards, {
        opacity: [0, 1],
        translateY: [16, 0],
        delay: stagger(40),
        duration: 420,
        ease: "outCubic",
      });
    }
  }, [dept]);

  const deptCount = Object.keys(s.departments || {}).length;

  return (
    <div className="flex min-h-0 w-full max-w-[1080px] flex-1 flex-col mx-auto px-6 pt-5 max-[640px]:px-3.5 max-[640px]:pt-3.5">
      <div className="mb-3.5 grid flex-none grid-cols-3 gap-3 max-[640px]:grid-cols-1">
        <div className="stat-card"><strong>{(s.total_qa || 0).toLocaleString()}</strong><span>问答对总量</span></div>
        <div className="stat-card"><strong>{deptCount || 16}</strong><span>覆盖科室</span></div>
        <div className="stat-card"><strong>{s.avg_answer_len || 144} 字</strong><span>平均答案长度</span></div>
      </div>

      <div className="flex flex-none items-center gap-3.5 max-[640px]:flex-wrap">
        <div className="kb-search">
          <Icon.search />
          <input
            type="search"
            value={query}
            placeholder="搜索样例问题、答案或疾病…"
            aria-label="搜索知识库样例"
            onChange={(e) => setQuery(e.target.value)}
          />
          {query && (
            <button className="icon-btn" onClick={() => setQuery("")} aria-label="清空搜索"><Icon.close /></button>
          )}
        </div>
        <span className="flex-none text-[12.5px] text-fg-muted">样例 {filtered.length} 条 · 完整库请直接在问答中检索</span>
      </div>

      <div className="flex flex-none flex-wrap gap-2 py-3.5" role="tablist" aria-label="科室筛选">
        <button className={"filter-chip" + (dept === "all" ? " active" : "")} onClick={() => setDept("all")}>全部</button>
        {DEPARTMENTS.map((c) => (
          <button
            key={c.id}
            className={"filter-chip hue-" + c.hue + (dept === c.name ? " active" : "")}
            onClick={() => setDept(c.name)}
          >
            {c.name}
          </button>
        ))}
      </div>

      <div className="kb-scroll min-h-0 flex-1 overflow-y-auto pb-6">
        {filtered.length === 0 ? (
          <div className="flex flex-col items-center gap-2 px-5 py-[70px] text-center text-fg-muted [&_svg]:size-10 [&_svg]:opacity-50">
            <Icon.search />
            <p className="text-[15px] font-semibold text-fg-soft">没有匹配「{query}」的样例</p>
            <span className="text-[13px]">试试更短的关键词，或切换到「全部」科室。</span>
            <button className="chip mt-2" onClick={() => { setQuery(""); setDept("all"); }}>重置筛选</button>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3.5 max-[960px]:grid-cols-1" ref={gridRef}>
            {filtered.map((e) => {
              const c = DEPARTMENTS.find((d) => d.name === e.label);
              return (
                <article key={e.id} className={"kb-card hue-" + (c ? c.hue : "sky")}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="kb-tag">{e.label}</span>
                    <span className="inline-flex items-center gap-1 text-[11.3px] text-fg-muted [&_svg]:size-3"><Icon.check /> 质量评分 {e.score}/5</span>
                  </div>
                  <h3 className="text-[15px] font-bold leading-[1.45]">{e.question}</h3>
                  <p className="flex-1 text-[12.8px] text-fg-soft">{e.answer.slice(0, 70)}…</p>
                  <div className="mt-1 flex items-center gap-2">
                    <button className="kb-link" onClick={() => onOpenEntry(e)}>
                      <Icon.doc /> 查看完整问答
                    </button>
                    <button className="kb-ask" onClick={() => onAsk(e.question)}>
                      <Icon.chat /> 就此提问
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}