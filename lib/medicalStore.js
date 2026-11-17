// 会话持久化客户端（与旧 lib/hrStore.js 完全一致，仅文件更名）

// API 基址由 NEXT_PUBLIC_API_URL 配置（见 lib/config.js）；部署时须与前端同站。
// 旧：const apiBase = () => `http://${window.location.hostname}:8000`;（跟随 hostname 写死端口）
import { API_BASE } from "./config.js";

async function req(path, { method = "GET", body } = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `后端返回 ${res.status}`;
    try {
      const j = await res.json();
      if (j && j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* 非 JSON 响应保留状态码描述 */ }
    throw new Error(detail);
  }
  return res.json();
}

export const fetchConversations = () => req("/api/conversations");
export const createConversation = (payload) => req("/api/conversations", { method: "POST", body: payload });
export const patchConversation = (id, patch) => req(`/api/conversations/${id}`, { method: "PATCH", body: patch });
export const deleteConversation = (id) => req(`/api/conversations/${id}`, { method: "DELETE" });
export const appendMessage = (convId, message) => req(`/api/conversations/${convId}/messages`, { method: "POST", body: message });
export const patchMessage = (msgId, patch) => req(`/api/messages/${msgId}`, { method: "PATCH", body: patch });