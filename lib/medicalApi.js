// 医疗前端后端客户端（改自 lib/hrApi.js）：问答走 FastAPI SSE 流式接口。
// 旧：askBackend 一次性 POST /api/ask 等完整 JSON（保留为非流式兜底）。
// API 基址由 NEXT_PUBLIC_API_URL 配置（见 lib/config.js）；部署时须与前端同站。
// 旧：const apiBase = () => `http://${window.location.hostname}:8000`;（跟随 hostname 写死端口）
import { API_BASE } from "./config.js";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/api/health`, { credentials: "include" });
  if (!res.ok) throw new Error(`后端返回 ${res.status}`);
  return res.json();
}

// 非流式兜底（SSE 不可用时回退）
export async function askBackend(question, { timeoutMs = 90000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/api/ask`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      let detail = `后端返回 ${res.status}`;
      try {
        const j = await res.json();
        if (j && j.detail) detail = j.detail;
      } catch { /* 非 JSON 响应保留状态码描述 */ }
      throw new Error(detail);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

// SSE 流式问答：meta(溯源/科室/疾病) -> delta(文本增量) -> done/error
export async function askBackendStream(question, { onMeta, onDelta, timeoutMs = 120000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/api/ask/stream`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: ctrl.signal,
    });
    if (!res.ok || !res.body) {
      let detail = `后端返回 ${res.status}`;
      try {
        const j = await res.json();
        if (j && j.detail) detail = j.detail;
      } catch { /* 非 JSON 响应保留状态码描述 */ }
      throw new Error(detail);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let meta = null;
    let text = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // SSE 事件以空行分隔
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let event = "message";
        let data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7).trim();
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (!data) continue;
        const payload = JSON.parse(data);
        if (event === "meta") {
          meta = payload;
          if (onMeta) onMeta(payload);
        } else if (event === "delta") {
          text += payload.text || "";
          if (onDelta) onDelta(payload.text || "", text);
        } else if (event === "error") {
          throw new Error(payload.detail || "流式问答失败");
        }
      }
    }
    return { ...(meta || {}), answer: text };
  } finally {
    clearTimeout(timer);
  }
}