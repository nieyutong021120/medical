"use client";

// 消息气泡（改自 components/hr/ChatMessage.jsx）：
// - 新增 SSE 真流式渲染（msg.streaming 时文本随 delta 增长，不再用本地打字机模拟）
// - 新增科室徽章(msg.department)与相关疾病标签(msg.related_diseases)
// - 溯源从"法条链接"改为"参考QA卡片"（点击打开抽屉）
// 旧：points/contact/suggestions/confidence/related 等HR静态知识分支已移除
import { useEffect, useRef, useState } from "react";
import { animate } from "animejs";
import Icon from "./Icons.jsx";

function fmtTime(ts) {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export default function ChatMessage({ msg, onCopy, onFeedback, onOpenSource, onAsk, onRevealed, onStream }) {
  const cardRef = useRef(null);
  const isBot = msg.role === "assistant";
  const typing = isBot && msg.fresh;
  const [shown, setShown] = useState(typing ? 0 : (msg.text || "").length);

  useEffect(() => {
    if (cardRef.current) {
      animate(cardRef.current, {
        opacity: [0, 1],
        translateY: [14, 0],
        duration: 420,
        ease: "outCubic",
      });
    }
  }, []);

  // SSE 流式：文本随 delta 增长，shown 同步到最新长度
  useEffect(() => {
    if (!typing) {
      setShown((msg.text || "").length);
      if (msg.streaming && onStream) onStream();
    }
  }, [msg.text, typing, msg.streaming]);

  // 本地打字机（仅 fresh 消息，如非流式兜底返回）
  useEffect(() => {
    if (!typing) return;
    const full = msg.text || "";
    const start = performance.now();
    const duration = Math.max(320, full.length * 8);
    const timer = setInterval(() => {
      const elapsed = performance.now() - start;
      const i = Math.min(full.length, Math.ceil((elapsed / duration) * full.length));
      setShown(i);
      if (onStream) onStream();
      if (i >= full.length) {
        clearInterval(timer);
        if (onRevealed) onRevealed(msg.id);
      }
    }, 16);
    return () => clearInterval(timer);
  }, [typing, msg.id]);

  if (!isBot) {
    return (
      <div className="flex flex-col items-end gap-[5px]" ref={cardRef}>
        <div className="bubble-user">{msg.text}</div>
        <span className="pr-1 text-[11px] text-fg-muted">{fmtTime(msg.ts)}</span>
      </div>
    );
  }

  if (msg.kind === "error") {
    return (
      <div className="flex gap-3" ref={cardRef}>
        <div className="bot-avatar"><Icon.bot /></div>
        <div className="min-w-0 flex-1">
          <div className="bubble-bot">
            <div className="bot-note warn"><Icon.info /> <span>{msg.text}</span></div>
          </div>
          <div className="mt-2 flex items-center gap-1 pl-0.5">
            <span className="ml-auto pr-1 text-[11px] text-fg-muted">{fmtTime(msg.ts)}</span>
          </div>
        </div>
      </div>
    );
  }

  const revealed = shown >= (msg.text || "").length && !msg.streaming;
  const tailDelay = Math.min((msg.text || "").length * 16, 1400);

  return (
    <div className="flex gap-3" ref={cardRef}>
      <div className="bot-avatar"><Icon.bot /></div>
      <div className="min-w-0 flex-1">
        <div className="bubble-bot">
          {revealed && msg.department && (
            <div className="mb-2 flex flex-wrap items-center gap-1.5">
              <span className="dept-badge"><Icon.cross /> 建议科室：{msg.department}</span>
              {(msg.related_diseases || []).map((d) => (
                <span key={d} className="disease-tag">{d}</span>
              ))}
            </div>
          )}

          {revealed && msg.rewrittenQuery && (
            <div className="mb-1.5 text-[12px] text-fg-muted">已理解为：{msg.rewrittenQuery}</div>
          )}

          <p className={"text-[14.5px] [overflow-wrap:anywhere]" + (msg.kind === "rag" ? " [white-space:pre-line]" : "")}>
            {(msg.text || "").slice(0, shown)}
            {(typing && !revealed) || msg.streaming ? <span className="caret" /> : null}
          </p>

          {revealed && msg.note && (
            <div className="bot-note" style={{ animationDelay: `${tailDelay * 0.35}ms` }}>
              <Icon.info /> <span>{msg.note}</span>
            </div>
          )}

          {revealed && msg.sources && msg.sources.length > 0 && (
            <div className="bot-sources" style={{ animationDelay: `${tailDelay * 0.5}ms` }}>
              <span className="inline-flex items-center gap-[5px] self-start text-xs text-fg-muted [&_svg]:size-[14px]"><Icon.doc /> 参考问答（{msg.sources.length}）</span>
              {/* 旧：chip 直接作为 bot-sources 的 flex 子项（宽度随文本、换行错乱）；现套等宽网格容器 */}
              <div className="source-list">
                {msg.sources.map((s, i) => (
                  <button key={s.id || i} className="source-chip" title={s.answer} onClick={() => onOpenSource(s)}>
                    <span className="min-w-0 flex-1 truncate">{s.question}</span>
                    <span className="src-label">{s.label}</span>
                    <Icon.chevron />
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {revealed && (
          <div className="mt-2 flex items-center gap-1 pl-0.5">
            <button className="action-btn" onClick={() => onCopy(msg)}><Icon.copy /> 复制</button>
            <button
              className={"action-btn" + (msg.feedback === "up" ? " on-up" : "")}
              onClick={() => onFeedback(msg.id, "up")}
              aria-pressed={msg.feedback === "up"}
            ><Icon.thumbUp /> 有帮助</button>
            <button
              className={"action-btn" + (msg.feedback === "down" ? " on-down" : "")}
              onClick={() => onFeedback(msg.id, "down")}
              aria-pressed={msg.feedback === "down"}
            ><Icon.thumbDown /> 待改进</button>
            <span className="ml-auto pr-1 text-[11px] text-fg-muted">{fmtTime(msg.ts)}</span>
          </div>
        )}
      </div>
    </div>
  );
}