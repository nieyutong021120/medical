// 前端 API 基址（唯一来源）：由 NEXT_PUBLIC_API_URL 注入。
// Next 静态导出(output:'export')在构建期把它固化进 JS，改地址后需重新 npm run build。
// 旧：部署时须设为与"访问前端所用 host"同站的地址，否则 SameSite=Lax 的 client_id Cookie 不生效。
// 现改用 JWT（Authorization 头），不再受同站限制，只需后端 CORS 白名单包含前端来源。
export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";