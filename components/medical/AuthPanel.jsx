"use client";

// 登录/注册面板：替代旧 Cookie 匿名身份，JWT 认证入口。
import { useState } from "react";
import { login, register } from "../../lib/auth.js";

export default function AuthPanel({ onAuthed }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (busy) return;
    setError("");
    setBusy(true);
    try {
      const user = mode === "login"
        ? await login(username.trim(), password)
        : await register(username.trim(), password);
      onAuthed(user);
    } catch (err) {
      setError(err.message || "网络错误，请确认后端已启动");
    } finally {
      setBusy(false);
    }
  };

  const switchMode = () => {
    setMode(mode === "login" ? "register" : "login");
    setError("");
  };

  return (
    <div className="hr-app hr-boot">
      <div className="w-[min(400px,calc(100vw-40px))] rounded-card bg-surface p-8 shadow-card">
        <div className="mb-1 text-center text-[22px] font-bold text-fg">华佗医疗智能助手</div>
        <p className="mb-6 text-center text-[13px] text-fg-muted">
          {mode === "login" ? "登录后继续你的健康咨询" : "创建账号，开始健康咨询"}
        </p>
        <form onSubmit={submit} className="flex flex-col gap-3.5">
          <label className="flex flex-col gap-1.5 text-[13px] text-fg-soft">
            用户名
            <input
              className="rounded-soft border border-line bg-canvas px-3.5 py-2.5 text-[14px] text-fg outline-none transition-colors focus:border-brand"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="2-32 个字符"
              autoComplete="username"
              autoFocus
              required
            />
          </label>
          <label className="flex flex-col gap-1.5 text-[13px] text-fg-soft">
            密码
            <input
              className="rounded-soft border border-line bg-canvas px-3.5 py-2.5 text-[14px] text-fg outline-none transition-colors focus:border-brand"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="至少 6 位"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              required
            />
          </label>
          {error && <div className="text-[13px] text-danger">{error}</div>}
          <button className="primary-btn mt-1 w-full disabled:opacity-60" type="submit" disabled={busy}>
            {busy ? "请稍候…" : mode === "login" ? "登 录" : "注册并登录"}
          </button>
        </form>
        <div className="mt-5 text-center text-[13px] text-fg-muted">
          {mode === "login" ? "还没有账号？" : "已有账号？"}
          <button className="ml-1 font-semibold text-brand-deep hover:underline" type="button" onClick={switchMode}>
            {mode === "login" ? "立即注册" : "去登录"}
          </button>
        </div>
      </div>
    </div>
  );
}