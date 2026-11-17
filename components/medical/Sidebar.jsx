"use client";

// 侧边导航（改自 components/hr/Sidebar.jsx，仅品牌与文案换医疗主题；CSS类名沿用）
import { useEffect, useRef } from "react";
import { animate, stagger } from "animejs";
import Icon from "./Icons.jsx";
import { ASSISTANT } from "../../data/medicalKnowledge.js";

function relTime(ts) {
  const days = Math.floor((Date.now() - ts) / 86400000);
  if (days <= 0) return "今天";
  if (days === 1) return "昨天";
  if (days < 30) return `${days} 天前`;
  return new Date(ts).toLocaleDateString("zh-CN");
}

export default function Sidebar({
  view, conversations, activeId, open,
  onNav, onNew, onSelect, onDelete, onClose,
}) {
  const listRef = useRef(null);

  useEffect(() => {
    if (!listRef.current) return;
    const items = listRef.current.querySelectorAll(".side-conv");
    if (items.length) {
      animate(items, {
        opacity: [0, 1],
        translateX: [-12, 0],
        delay: stagger(50),
        duration: 400,
        ease: "outCubic",
      });
    }
  }, []);

  return (
    <aside className={"hr-sidebar" + (open ? " open" : "")} aria-label="侧边导航">
      <div className="flex items-center gap-[11px] px-1 py-0.5">
        <div className="brand-mark"><Icon.cross /></div>
        <div className="flex min-w-0 flex-col leading-[1.3]">
          <strong className="text-[15.5px] tracking-[0.2px]">{ASSISTANT.name}</strong>
          <span className="text-[11.5px] text-[rgba(232,235,244,0.55)]">{ASSISTANT.dataset} · 健康咨询</span>
        </div>
        <button className="icon-btn side-close" onClick={onClose} aria-label="收起侧栏">
          <Icon.close />
        </button>
      </div>

      <button className="new-chat-btn" onClick={onNew}>
        <Icon.plus /> 新建对话
      </button>

      <nav className="flex flex-col gap-1">
        <button className={"side-nav-item" + (view === "chat" ? " active" : "")} onClick={() => onNav("chat")}>
          <Icon.chat /> 智能问答
        </button>
        <button className={"side-nav-item" + (view === "knowledge" ? " active" : "")} onClick={() => onNav("knowledge")}>
          <Icon.book /> 知识库概览
        </button>
      </nav>

      <div className="flex items-center gap-[7px] px-3 pt-1.5 text-[11.5px] uppercase tracking-[1.2px] text-[rgba(232,235,244,0.42)] [&_svg]:size-[13px]">
        <Icon.clock /> 对话历史
      </div>
      <div className="side-convs flex min-h-0 flex-1 flex-col gap-[3px] overflow-y-auto p-0.5" ref={listRef}>
        {conversations.length === 0 && (
          <p className="px-3 py-2.5 text-[12.5px] leading-[1.7] text-[rgba(232,235,244,0.4)]">还没有对话，点上方「新建对话」开始提问。</p>
        )}
        {conversations.map((c) => (
          <div
            key={c.id}
            className={"side-conv" + (c.id === activeId && view === "chat" ? " active" : "")}
            onClick={() => onSelect(c.id)}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(c.id); } }}
            role="button"
            tabIndex={0}
          >
            <div className="flex min-w-0 flex-1 flex-col leading-[1.35]">
              <span className="truncate text-[13.5px] text-[#e8ebf4]">{c.title}</span>
              <span className="text-[11px] text-[rgba(232,235,244,0.42)]">{relTime(c.createdAt)} · {c.messages.filter((m) => m.role === "user").length} 问</span>
            </div>
            <button
              className="icon-btn conv-delete"
              aria-label={`删除对话「${c.title}」`}
              onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
            >
              <Icon.trash />
            </button>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2.5 border-t border-[rgba(255,255,255,0.08)] px-2 pb-1 pt-3">
        <div className="user-avatar">{ASSISTANT.user.name.slice(0, 1)}</div>
        <div className="flex min-w-0 flex-1 flex-col leading-[1.3]">
          <strong className="text-[13.5px]">{ASSISTANT.user.name}</strong>
          <span className="text-[11.5px] text-[rgba(232,235,244,0.5)]">{ASSISTANT.user.dept} · {ASSISTANT.user.role}</span>
        </div>
        <span className="size-[9px] flex-none rounded-full bg-ok shadow-[0_0_0_3px_rgba(16,185,129,0.22)]" title="在线" />
      </div>
    </aside>
  );
}