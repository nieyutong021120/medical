"use client";

// 聊天面板（改自 components/hr/ChatPanel.jsx）：欢迎页按科室展示真实样例问题
import { useEffect, useRef, useState } from "react";
import { animate, stagger } from "animejs";
import Icon from "./Icons.jsx";
import ChatMessage from "./ChatMessage.jsx";
import { ASSISTANT, DEPARTMENTS, QUICK_QUESTIONS, FALLBACK_STATS } from "../../data/medicalKnowledge.js";

const MAX_LEN = 500;

function Welcome({ onAsk, stats }) {
  const ref = useRef(null);
  const total = (stats && stats.total_qa) || FALLBACK_STATS.total_qa;
  useEffect(() => {
    if (!ref.current) return;
    const cards = ref.current.querySelectorAll(".welcome-card");
    animate(cards, {
      opacity: [0, 1],
      translateY: [22, 0],
      delay: stagger(80),
      duration: 600,
      ease: "outCubic",
    });
  }, []);

  return (
    <div className="mx-auto max-w-[920px] px-6 pb-6 pt-10 max-[640px]:px-3.5 max-[640px]:pb-4 max-[640px]:pt-[26px]" ref={ref}>
      <div className="mb-[26px] text-left">
        <div className="welcome-badge"><Icon.pulse /> 医疗问答 · 有据可查</div>
        <h1 className="text-[clamp(24px,3.4vw,32px)] font-extrabold leading-[1.35] tracking-[0.2px]">你好，我是{ASSISTANT.name}<span className="wave">。</span></h1>
        <p className="mt-2 max-w-[640px] text-sm text-fg-soft">基于华佗开源医疗问答库 {total.toLocaleString()} 条真实问答，回答附参考溯源与建议科室。输入健康问题，或从下面的科室开始探索。</p>
      </div>

      <div className="grid grid-cols-3 gap-3.5 max-[1100px]:grid-cols-2 max-[640px]:grid-cols-1">
        {DEPARTMENTS.map((c) => {
          const count = stats && stats.departments ? stats.departments[c.name] : FALLBACK_STATS.departments[c.name];
          const DeptIcon = Icon[c.icon] || Icon.cross;
          return (
            <button key={c.id} className={"welcome-card hue-" + c.hue} onClick={() => onAsk(c.sample)}>
              <span className="wc-icon"><DeptIcon /></span>
              <span className="flex min-w-0 flex-1 flex-col gap-[3px]">
                <strong className="text-[14.5px]">{c.name}</strong>
                <span className="text-[11.8px] text-fg-muted">{c.blurb}</span>
                <span className="wc-sample">{c.sample}</span>
              </span>
              {count ? <span className="wc-count">{count.toLocaleString()} 条</span> : null}
            </button>
          );
        })}
      </div>

      <div className="mt-6">
        <span className="mb-2.5 inline-flex items-center gap-1.5 text-[12.5px] font-semibold text-fg-soft [&_svg]:size-[15px] [&_svg]:text-brand"><Icon.trend /> 热门问题</span>
        <div className="flex flex-wrap gap-2">
          {QUICK_QUESTIONS.map((q) => (
            <button key={q} className="chip" onClick={() => onAsk(q)}>{q}</button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function ChatPanel({
  messages, pending, stats, onSend, onCopy, onFeedback, onOpenSource, onAsk, onRevealed,
}) {
  const scrollRef = useRef(null);
  const taRef = useRef(null);
  const [draft, setDraft] = useState("");

  const scrollToBottom = (smooth = true) => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  };

  useEffect(() => { scrollToBottom(); }, [messages.length, pending]);

  const submit = () => {
    const text = draft.trim();
    if (!text || pending) return;
    setDraft("");
    if (taRef.current) taRef.current.style.height = "auto";
    onSend(text);
  };

  const autoGrow = (el) => {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  const empty = messages.length === 0 && !pending;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="chat-scroll min-h-0 flex-1 overflow-y-auto" ref={scrollRef}>
        {empty ? (
          <Welcome onAsk={onAsk} stats={stats} />
        ) : (
          <div className="mx-auto flex max-w-[880px] flex-col gap-[22px] px-6 pb-3 pt-[26px] max-[640px]:px-3.5 max-[640px]:pb-2 max-[640px]:pt-[18px]">
            {messages.map((m) => (
              <ChatMessage
                key={m.id}
                msg={m}
                onCopy={onCopy}
                onFeedback={onFeedback}
                onOpenSource={onOpenSource}
                onAsk={onAsk}
                onRevealed={onRevealed}
                onStream={() => scrollToBottom(false)}
              />
            ))}
            {pending && (
              <div className="flex gap-3">
                <div className="bot-avatar"><Icon.bot /></div>
                <div className="bubble-bot typing">
                  <span className="text-[13px] text-fg-muted">正在检索医疗知识库</span>
                  <span className="dots"><i /><i /><i /></span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mx-auto w-full max-w-[928px] flex-none px-6 pb-3.5 pt-2.5 max-[640px]:px-3.5 max-[640px]:pb-2.5 max-[640px]:pt-2">
        <div className="composer-box">
          <textarea
            ref={taRef}
            rows={1}
            maxLength={MAX_LEN}
            value={draft}
            placeholder="输入健康问题，例如：孩子发烧38度怎么办？"
            aria-label="问题输入框"
            onChange={(e) => { setDraft(e.target.value); autoGrow(e.target); }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
          />
          <div className="mt-1.5 flex items-center gap-3">
            <span className="text-[11.5px] text-fg-muted max-[640px]:hidden">Enter 发送 · Shift+Enter 换行</span>
            <span className={"ml-auto text-[11.5px] " + (draft.length >= MAX_LEN ? "font-semibold text-danger" : "text-fg-muted")}>{draft.length}/{MAX_LEN}</span>
            <button className="send-btn" onClick={submit} disabled={!draft.trim() || pending} aria-label="发送">
              <Icon.send />
            </button>
          </div>
        </div>
        <p className="mt-2 text-center text-[11.3px] text-fg-muted max-[640px]:text-[10.5px]">回答基于华佗开源医疗问答库检索生成，仅供健康参考，不能替代执业医师诊断与治疗建议。</p>
      </div>
    </div>
  );
}