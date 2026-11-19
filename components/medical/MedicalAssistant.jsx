"use client";

// 医疗助手主组件（改自 components/hr/HRAssistant.jsx）：
// - 问答改为 SSE 真流式（meta 先到→渲染溯源/科室，delta 逐步填充回答）
// - 顶栏状态与欢迎页统计来自 /api/health 实测
// 会话管理（后端SQLite+JWT用户身份、乐观渲染、撤销删除）逻辑原样沿用。
// 旧：身份为 Cookie 匿名 client_id；现改 JWT 登录认证（lib/auth.js + AuthPanel）。
import { useCallback, useEffect, useRef, useState } from "react";
import Icon from "./Icons.jsx";
import Sidebar from "./Sidebar.jsx";
import ChatPanel from "./ChatPanel.jsx";
import KnowledgePanel from "./KnowledgePanel.jsx";
import SourceDrawer from "./SourceDrawer.jsx";
import AuthPanel from "./AuthPanel.jsx";
import { ASSISTANT, FALLBACK_STATS } from "../../data/medicalKnowledge.js";
// 旧：import { askBackend, askBackendStream, fetchHealth }（askBackend 为 SSE 失败回退，已弃用）
import { askBackendStream, fetchHealth } from "../../lib/medicalApi.js";
import { API_BASE } from "../../lib/config.js";
import { getToken, fetchMe, logout as authLogout } from "../../lib/auth.js";
import {
  fetchConversations, createConversation, patchConversation,
  deleteConversation as apiDeleteConversation, appendMessage, patchMessage,
} from "../../lib/medicalStore.js";

// 临时本地 key：仅用于乐观渲染与后端离线时的内存会话；持久化 id 一律由后端 uid() 生成
let localSeq = 0;
const tmpId = () => `local-${++localSeq}`;

export default function MedicalAssistant() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [view, setView] = useState("chat");
  const [pending, setPending] = useState(false);
  const [drawer, setDrawer] = useState(null);
  const [toast, setToast] = useState(null);
  const [sideOpen, setSideOpen] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [stats, setStats] = useState(null);
  const [online, setOnline] = useState(false);
  // undefined=校验 token 中；null=未登录（渲染 AuthPanel）；对象=已登录用户
  const [user, setUser] = useState(undefined);

  const toastTimer = useRef(null);
  const bootRef = useRef(false);

  const showToast = useCallback((text, action) => {
    setToast({ text, action });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), action ? 6000 : 3200);
  }, []);

  // 旧：首次挂载直接按 Cookie 身份加载历史（无登录概念）；现拆为"身份校验"+"登录后加载"两步。
  // useEffect(() => {
  //   if (!bootRef.current) {
  //     bootRef.current = true;
  //     (async () => {
  //       try {
  //         const list = await fetchConversations();
  //         const nonEmpty = list.filter((c) => c.messages.length > 0);
  //         let empty = list.find((c) => c.messages.length === 0);
  //         if (!empty) {
  //           empty = await createConversation({ title: "新对话", createdAt: Date.now() });
  //         }
  //         setConversations([empty, ...nonEmpty]);
  //         setActiveId(empty.id);
  //       } catch {
  //         const fresh = { id: tmpId(), title: "新对话", createdAt: Date.now(), messages: [] };
  //         setConversations([fresh]);
  //         setActiveId(fresh.id);
  //         showToast("无法加载对话历史（后端未启动）");
  //       } finally {
  //         setHydrated(true);
  //       }
  //     })();
  //     fetchHealth()
  //       .then((h) => { setStats(h.stats); setOnline(true); })
  //       .catch(() => { setStats(FALLBACK_STATS); setOnline(false); });
  //   }
  //   return () => {
  //     if (toastTimer.current) clearTimeout(toastTimer.current);
  //   };
  // }, [showToast]);

  // 挂载：有 token 则向 /api/auth/me 校验，无/失效进登录页；健康检查与登录无关，照常拉取。
  useEffect(() => {
    if (!getToken()) {
      setUser(null);
    } else {
      fetchMe()
        .then((u) => setUser(u))
        .catch(() => setUser(null));
    }
    fetchHealth()
      .then((h) => { setStats(h.stats); setOnline(true); })
      .catch(() => { setStats(FALLBACK_STATS); setOnline(false); });
    return () => {
      if (toastTimer.current) clearTimeout(toastTimer.current);
    };
  }, []);

  // 登录成功后：从后端 SQLite 加载历史（按用户 id 隔离）。
  // 复用已有空会话，无则新建一个。bootRef 防 StrictMode 双调用重复建会话。
  useEffect(() => {
    if (!user || bootRef.current) return;
    bootRef.current = true;
    (async () => {
      try {
        const list = await fetchConversations();
        const nonEmpty = list.filter((c) => c.messages.length > 0);
        let empty = list.find((c) => c.messages.length === 0);
        if (!empty) {
          empty = await createConversation({ title: "新对话", createdAt: Date.now() });
        }
        setConversations([empty, ...nonEmpty]);
        setActiveId(empty.id);
      } catch {
        const fresh = { id: tmpId(), title: "新对话", createdAt: Date.now(), messages: [] };
        setConversations([fresh]);
        setActiveId(fresh.id);
        showToast("无法加载对话历史（后端未启动）");
      } finally {
        setHydrated(true);
      }
    })();
  }, [user, showToast]);

  const resetToLogin = useCallback(() => {
    setUser(null);
    bootRef.current = false;
    setHydrated(false);
    setConversations([]);
    setActiveId(null);
  }, []);

  // 任一请求 401（token 过期/失效）：lib 层已清 token，这里退回登录页
  useEffect(() => {
    window.addEventListener("medical:unauthorized", resetToLogin);
    return () => window.removeEventListener("medical:unauthorized", resetToLogin);
  }, [resetToLogin]);

  const logout = useCallback(() => {
    authLogout();
    resetToLogin();
  }, [resetToLogin]);

  const active = conversations.find((c) => c.id === activeId) || null;

  const newConversation = useCallback(async () => {
    const current = conversations.find((c) => c.id === activeId);
    if (current && current.messages.length === 0) {
      setView("chat");
      setSideOpen(false);
      return;
    }
    let fresh;
    try {
      fresh = await createConversation({ title: "新对话", createdAt: Date.now() });
    } catch {
      fresh = { id: tmpId(), title: "新对话", createdAt: Date.now(), messages: [] };
      showToast("后端未启动，新对话暂存于内存");
    }
    setConversations((prev) => [fresh, ...prev]);
    setActiveId(fresh.id);
    setView("chat");
    setSideOpen(false);
  }, [conversations, activeId, showToast]);

  const selectConversation = useCallback((id) => {
    setActiveId(id);
    setView("chat");
    setSideOpen(false);
  }, []);

  const deleteConversation = useCallback(async (id) => {
    const idx = conversations.findIndex((c) => c.id === id);
    if (idx < 0) return;
    const removed = conversations[idx];
    let next = conversations.filter((c) => c.id !== id);
    if (id === activeId) {
      if (next.length) {
        setActiveId(next[0].id);
      } else {
        let fresh;
        try {
          fresh = await createConversation({ title: "新对话", createdAt: Date.now() });
        } catch {
          fresh = { id: tmpId(), title: "新对话", createdAt: Date.now(), messages: [] };
        }
        next = [fresh];
        setActiveId(fresh.id);
      }
    }
    setConversations(next);
    apiDeleteConversation(id).catch(() => showToast("删除失败（后端未启动）"));
    showToast(`已删除「${removed.title}」`, {
      label: "撤销",
      fn: () => {
        const restore = (conv) => setConversations((cur) => {
          const at = Math.min(idx, cur.length);
          const arr = [...cur];
          arr.splice(at, 0, conv);
          return arr;
        });
        createConversation({ title: removed.title, createdAt: removed.createdAt, messages: removed.messages })
          .then((restored) => restore(restored || removed))
          .catch(() => restore(removed));
      },
    });
  }, [conversations, activeId, showToast]);

  // 消息落库后后端返回真实 id，替换乐观渲染用的临时 key
  const swapMsgId = useCallback((convId, oldId, newId) => {
    setConversations((prev) => prev.map((c) => (
      c.id === convId
        ? { ...c, messages: c.messages.map((m) => (m.id === oldId ? { ...m, id: newId } : m)) }
        : c
    )));
  }, []);

  const patchMsg = useCallback((convId, msgId, fn) => {
    setConversations((prev) => prev.map((c) => (
      c.id === convId ? { ...c, messages: c.messages.map((m) => (m.id === msgId ? fn(m) : m)) } : c
    )));
  }, []);

  const send = useCallback((raw) => {
    const text = raw.trim();
    if (!text || pending) return;
    const convId = activeId;
    const conv = conversations.find((c) => c.id === convId);
    const firstMessage = !conv || conv.messages.length === 0;
    const title = firstMessage ? text.slice(0, 18) + (text.length > 18 ? "…" : "") : conv.title;
    const userMsg = { id: tmpId(), role: "user", text, ts: Date.now() };
    setConversations((prev) => prev.map((c) => (
      c.id === convId ? { ...c, title, messages: [...c.messages, userMsg] } : c
    )));
    appendMessage(convId, userMsg)
      .then((saved) => swapMsgId(convId, userMsg.id, saved.id))
      .catch(() => showToast("历史保存失败（后端未启动）"));
    if (firstMessage) patchConversation(convId, { title }).catch(() => {});
    setPending(true);

    // 旧：askBackend(text).then(pushBot)（一次性返回后本地打字机模拟；现改SSE真流式）
    const botId = tmpId();
    let metaSeen = false;
    let botMsg = null;
    const insertBot = (meta) => {
      botMsg = {
        id: botId, role: "assistant", kind: "rag", ts: Date.now(),
        text: "", streaming: true, fresh: false,
        sources: meta.sources || [],
        department: meta.department || "",
        related_diseases: meta.related_diseases || [],
        note: meta.disclaimer || "",
        route: meta.route,
      };
      setConversations((prev) => prev.map((c) => (
        c.id === convId ? { ...c, messages: [...c.messages, botMsg] } : c
      )));
    };
    const finishBot = (fullText) => {
      patchMsg(convId, botId, (m) => ({ ...m, text: fullText, streaming: false }));
      // 落库放在 updater 外：StrictMode 会双调用 updater，副作用放里面会重复入库
      if (botMsg) {
        const { streaming, fresh, ...toSave } = { ...botMsg, text: fullText };
        appendMessage(convId, toSave)
          .then((saved) => swapMsgId(convId, botId, saved.id))
          .catch(() => {});
      }
    };
    const failBot = (err) => {
      const detail = err.name === "AbortError" ? "请求超时" : (err.message || "网络错误");
      // 旧：所有失败统一提示"无法连接后端"；现区分限流/网络错误/问答失败
      // const errText = `无法连接医疗后端（${API_BASE}）：${detail}。请先启动后端服务后重试。`;
      const rateLimited = /429|rate ?limit|RPM|too many requests/i.test(detail);
      let errText;
      if (rateLimited) {
        errText = "请求过于频繁，请稍后再试（模型接口每分钟调用次数有限）。";
      } else if (err instanceof TypeError) {
        errText = `无法连接医疗后端（${API_BASE}）：${detail}。请先启动后端服务后重试。`;
      } else {
        errText = `问答失败：${detail}。请稍后重试。`;
      }
      if (metaSeen) {
        patchMsg(convId, botId, (m) => ({ ...m, kind: "error", text: errText, streaming: false }));
      } else {
        const botMsg = { id: botId, role: "assistant", kind: "error", text: errText, ts: Date.now() };
        setConversations((prev) => prev.map((c) => (
          c.id === convId ? { ...c, messages: [...c.messages, botMsg] } : c
        )));
      }
    };

    askBackendStream(text, {
      onMeta: (meta) => { metaSeen = true; setPending(false); insertBot(meta); },
      onDelta: (_d, full) => patchMsg(convId, botId, (m) => ({ ...m, text: full })),
    })
      .then((result) => finishBot(result.answer || ""))
      .catch((err) => {
        // 旧：SSE 未建立时回退一次性接口 /api/ask；会双倍消耗 LLM 调用（RPM=3 易 429），已弃用
        // if (!metaSeen) {
        //   try {
        //     const result = await askBackend(text);
        //     metaSeen = true;
        //     setPending(false);
        //     insertBot(result);
        //     finishBot(result.answer || "");
        //     return;
        //   } catch { /* 回退也失败，走错误气泡 */ }
        // }
        failBot(err);
      })
      .finally(() => setPending(false));
  }, [activeId, pending, conversations, showToast, swapMsgId, patchMsg]);

  const markRevealed = useCallback((msgId) => {
    setConversations((prev) => prev.map((c) => ({
      ...c,
      messages: c.messages.map((m) => (m.id === msgId ? { ...m, fresh: false } : m)),
    })));
  }, []);

  const feedback = useCallback((msgId, val) => {
    const msg = conversations.flatMap((c) => c.messages).find((m) => m.id === msgId);
    const next = msg && msg.feedback === val ? null : val;
    setConversations((prev) => prev.map((c) => ({
      ...c,
      messages: c.messages.map((m) => (m.id === msgId ? { ...m, feedback: next } : m)),
    })));
    patchMessage(msgId, { feedback: next }).catch(() => showToast("反馈保存失败（后端未启动）"));
    showToast(val === "up" ? "感谢反馈，很高兴帮到你" : "已记录，我们会持续优化知识库");
  }, [conversations, showToast]);

  const copyMessage = useCallback(async (msg) => {
    const parts = [msg.text];
    if (msg.department) parts.push("建议科室：" + msg.department);
    if (msg.note) parts.push("提示：" + msg.note);
    if (msg.sources) parts.push(...msg.sources.map((s) => `参考：${s.question}`));
    const text = parts.filter(Boolean).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      showToast("已复制回答到剪贴板");
    } catch {
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        showToast(ok ? "已复制回答到剪贴板" : "复制失败，请手动选择文本");
      } catch {
        showToast("复制失败，请手动选择文本");
      }
    }
  }, [showToast]);

  const askFromAnywhere = useCallback((question) => {
    setDrawer(null);
    setView("chat");
    setSideOpen(false);
    send(question);
  }, [send]);

  const openEntry = useCallback((entry) => setDrawer(entry), []);

  if (user === undefined) {
    return <div className="hr-app hr-boot"><div className="boot-spinner" /><span>正在加载医疗助手…</span></div>;
  }
  if (user === null) {
    return <AuthPanel onAuthed={setUser} />;
  }
  if (!hydrated) {
    return <div className="hr-app hr-boot"><div className="boot-spinner" /><span>正在加载医疗助手…</span></div>;
  }

  const total = (stats && stats.total_qa) || FALLBACK_STATS.total_qa;

  return (
    <div className="hr-app">
      <div className={"side-scrim" + (sideOpen ? " show" : "")} onClick={() => setSideOpen(false)} />
      <Sidebar
        view={view}
        conversations={conversations}
        activeId={activeId}
        open={sideOpen}
        onNav={(v) => { setView(v); setSideOpen(false); }}
        onNew={newConversation}
        onSelect={selectConversation}
        onDelete={deleteConversation}
        onClose={() => setSideOpen(false)}
      />

      <div className="hr-main">
        <header className="hr-topbar">
          <button className="icon-btn menu-btn" onClick={() => setSideOpen(true)} aria-label="打开侧栏"><Icon.menu /></button>
          <div className="min-w-0 flex-1 leading-[1.3]">
            <h2 className="text-[16.5px] font-bold">{view === "chat" ? "智能问答" : "知识库概览"}</h2>
            <span className="block truncate text-xs text-fg-muted max-[640px]:hidden">{view === "chat" ? (active ? active.title : "") : "华佗开源医疗问答库 · 统计与样例"}</span>
          </div>
          <div className="flex flex-none items-center gap-2.5">
            <span className="status-pill"><i className="pulse" /> {online ? `知识库在线 · ${total.toLocaleString()} 条问答` : "知识库离线 · 展示样例数据"}</span>
            <span className="env-badge">演示环境</span>
            <span className="max-[640px]:hidden text-xs font-semibold text-fg-soft">{user.username}</span>
            <button
              className="rounded-lg border border-line px-3 py-1.5 text-xs text-fg-soft transition-colors hover:border-fg-muted hover:text-fg"
              onClick={logout}
            >
              退出
            </button>
          </div>
        </header>

        {view === "chat" ? (
          <ChatPanel
            messages={active ? active.messages : []}
            pending={pending}
            stats={stats}
            onSend={send}
            onCopy={copyMessage}
            onFeedback={feedback}
            onOpenSource={openEntry}
            onAsk={askFromAnywhere}
            onRevealed={markRevealed}
          />
        ) : (
          <KnowledgePanel onOpenEntry={openEntry} onAsk={askFromAnywhere} stats={stats} />
        )}
      </div>

      {drawer && (
        <SourceDrawer
          entry={drawer}
          onClose={() => setDrawer(null)}
          onAsk={askFromAnywhere}
        />
      )}

      {toast && (
        <div className="toast" role="status">
          <Icon.check />
          <span>{toast.text}</span>
          {toast.action && (
            <button className="toast-action" onClick={() => { toast.action.fn(); setToast(null); }}>
              <Icon.refresh /> {toast.action.label}
            </button>
          )}
        </div>
      )}
    </div>
  );
}