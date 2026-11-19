// JWT 认证客户端：token 存 localStorage，请求统一带 Authorization: Bearer 头。
// 替代旧 Cookie（client_id httponly）匿名身份方案。
import { API_BASE } from "./config.js";

const TOKEN_KEY = "medical_token";

export const getToken = () =>
  typeof window === "undefined" ? null : localStorage.getItem(TOKEN_KEY);
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

export function authHeaders() {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

// 会话中收到 401（token 过期/失效）：清 token 并广播事件，主组件退回登录页
export function notifyUnauthorized() {
  clearToken();
  window.dispatchEvent(new CustomEvent("medical:unauthorized"));
}

async function authReq(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error((j && j.detail) || `后端返回 ${res.status}`);
  }
  return j;
}

export async function login(username, password) {
  const j = await authReq("/api/auth/login", { username, password });
  setToken(j.token);
  return j.user;
}

export async function register(username, password) {
  const j = await authReq("/api/auth/register", { username, password });
  setToken(j.token);
  return j.user;
}

// 启动时校验已存 token；失效则清掉并返回 null
export async function fetchMe() {
  const res = await fetch(`${API_BASE}/api/auth/me`, { headers: authHeaders() });
  if (res.status === 401) {
    clearToken();
    return null;
  }
  if (!res.ok) throw new Error(`后端返回 ${res.status}`);
  return res.json();
}

export function logout() {
  clearToken();
}