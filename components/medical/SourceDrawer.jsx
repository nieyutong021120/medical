"use client";

// 溯源抽屉（改自 components/hr/SourceDrawer.jsx）：展示参考QA完整内容
// entry 形态统一为 {question, answer, label, score, related_diseases}（样例与RAG命中同构）
import { useEffect } from "react";
import Icon from "./Icons.jsx";
import { DEPARTMENTS } from "../../data/medicalKnowledge.js";

export default function SourceDrawer({ entry, onClose, onAsk }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!entry) return null;
  const c = DEPARTMENTS.find((d) => d.name === entry.label);

  return (
    <div className="fixed inset-0 z-[60]">
      <div className="drawer-scrim" onClick={onClose} />
      <aside className={"drawer hue-" + (c ? c.hue : "sky")} role="dialog" aria-modal="true" aria-label={entry.question}>
        <header className="drawer-head">
          <div className="flex min-w-0 flex-1 flex-col gap-[7px]">
            <span className="kb-tag">{entry.label || "参考问答"}</span>
            <h2 className="text-[17px] leading-[1.4] [font-weight:750]">{entry.question}</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="关闭详情"><Icon.close /></button>
        </header>

        <div className="drawer-scroll min-h-0 flex-1 overflow-y-auto px-5 py-[18px]">
          <div className="mb-4 flex flex-wrap gap-3.5 text-xs text-fg-muted [&_svg]:size-[13px]">
            <span className="inline-flex items-center gap-[5px"><Icon.check /> 质量评分 {entry.score}/5</span>
            <span className="inline-flex items-center gap-[5px"><Icon.doc /> 来源：Huatuo-Lite</span>
            {entry.related_diseases && (
              <span className="inline-flex items-center gap-[5px"><Icon.pulse /> 相关疾病：{entry.related_diseases}</span>
            )}
          </div>

          <section className="drawer-block">
            <h4>医生回答</h4>
            <p className="text-[13.8px] leading-[1.85] text-fg-soft [white-space:pre-line]">{entry.answer}</p>
          </section>

          <section className="drawer-block">
            <h4>说明</h4>
            <div className="bot-note">
              <Icon.info />
              <span>该问答来自华佗开源医疗问答库（网络高频问题 + ChatGPT 改写答案），仅供健康参考，不能替代执业医师诊断。</span>
            </div>
          </section>
        </div>

        <footer className="flex flex-none gap-2.5 border-t border-line px-5 py-3.5">
          <button className="primary-btn flex-1" onClick={() => onAsk(entry.question)}>
            <Icon.chat /> 就此提问
          </button>
          <button className="ghost-btn" onClick={onClose}>关闭</button>
        </footer>
      </aside>
    </div>
  );
}