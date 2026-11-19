// app/layout.jsx 是 Next.js 的"全站外壳"——所有页面都套在它里面。
// 它提供 <html>/<body> 并加载全局样式；医疗助手自己管理全视口应用布局。
// 唯一 CSS 入口：Tailwind v4 + 设计令牌 + 保留的复杂视觉层。
import "./globals.css";

export const metadata = {
  title: "华佗医疗智能助手",
  description: "基于华佗开源医疗问答库（Huatuo-Lite）的智能问答：症状咨询、疾病知识、用药常识、就医科室推荐，回答附参考溯源",
};

export default function RootLayout({ children }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}