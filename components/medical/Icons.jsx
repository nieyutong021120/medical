// 轻量内联 SVG 图标集：不引第三方图标库，保持零新增依赖。
// 所有图标继承 currentColor，尺寸由 CSS 控制，方便在各处复用。
// （改自 components/hr/Icons.jsx：新增医疗图标 pulse/heart/baby/cross/eye，移除HR专用图标）

const base = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

export const Icon = {
  spark: (p) => (
    <svg {...base} {...p}><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z" /><path d="M18 15l.8 2.2L21 18l-2.2.8L18 21l-.8-2.2L15 18l2.2-.8z" /></svg>
  ),
  chat: (p) => (
    <svg {...base} {...p}><path d="M21 15a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2z" /><path d="M8 9h8M8 12.5h5" /></svg>
  ),
  book: (p) => (
    <svg {...base} {...p}><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z" /><path d="M4 5.5V20.5" /><path d="M9 7.5h7M9 11h5" /></svg>
  ),
  plus: (p) => (<svg {...base} {...p}><path d="M12 5v14M5 12h14" /></svg>),
  search: (p) => (<svg {...base} {...p}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.2-3.2" /></svg>),
  send: (p) => (<svg {...base} {...p}><path d="M4.5 12 20 4.5 15.5 20l-3.6-6.4z" /><path d="m11.9 13.6 8.1-9.1" /></svg>),
  close: (p) => (<svg {...base} {...p}><path d="M6 6l12 12M18 6 6 18" /></svg>),
  menu: (p) => (<svg {...base} {...p}><path d="M4 7h16M4 12h16M4 17h16" /></svg>),
  copy: (p) => (<svg {...base} {...p}><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h8" /></svg>),
  thumbUp: (p) => (<svg {...base} {...p}><path d="M7 11v9H4a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1z" /><path d="M7 11l4-7a2 2 0 0 1 2 2v3h5.5a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 17.3 20H7" /></svg>),
  thumbDown: (p) => (<svg {...base} {...p}><path d="M7 13V4H4a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1z" /><path d="M7 13l4 7a2 2 0 0 0 2-2v-3h5.5a2 2 0 0 0 2-2.3l-1.2-6A2 2 0 0 0 17.3 4H7" /></svg>),
  doc: (p) => (<svg {...base} {...p}><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /><path d="M9 13h6M9 16.5h4" /></svg>),
  chevron: (p) => (<svg {...base} {...p}><path d="m9 6 6 6-6 6" /></svg>),
  trash: (p) => (<svg {...base} {...p}><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13" /><path d="M10 11v6M14 11v6" /></svg>),
  check: (p) => (<svg {...base} {...p}><path d="m5 12.5 4.5 4.5L19 7" /></svg>),
  clock: (p) => (<svg {...base} {...p}><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></svg>),
  trend: (p) => (<svg {...base} {...p}><path d="M4 17l5-5 3 3 7-7" /><path d="M15 8h5v5" /></svg>),
  user: (p) => (<svg {...base} {...p}><circle cx="12" cy="8" r="3.5" /><path d="M5 20a7 7 0 0 1 14 0" /></svg>),
  bot: (p) => (<svg {...base} {...p}><rect x="4" y="7" width="16" height="12" rx="3" /><path d="M12 7V4M9 12.5h.01M15 12.5h.01" /><path d="M9.5 16h5" /><circle cx="12" cy="3.5" r="1" /></svg>),
  info: (p) => (<svg {...base} {...p}><circle cx="12" cy="12" r="8.5" /><path d="M12 11v5M12 8h.01" /></svg>),
  refresh: (p) => (<svg {...base} {...p}><path d="M20 11a8 8 0 0 0-14.9-3M4 13a8 8 0 0 0 14.9 3" /><path d="M4 5v4h4M20 19v-4h-4" /></svg>),
  shield: (p) => (<svg {...base} {...p}><path d="M12 3 5 6v5c0 4.5 3 8 7 10 4-2 7-5.5 7-10V6z" /><path d="m9 12 2 2 4-4" /></svg>),
  // ---- 医疗主题新增 ----
  pulse: (p) => (<svg {...base} {...p}><path d="M3 12h4l2-5 3 10 2.5-5H21" /></svg>),
  heart: (p) => (<svg {...base} {...p}><path d="M12 20s-7-4.6-9-9a5 5 0 0 1 9-3 5 5 0 0 1 9 3c-2 4.4-9 9-9 9z" /></svg>),
  baby: (p) => (<svg {...base} {...p}><circle cx="12" cy="9" r="5.5" /><path d="M9.5 8.5h.01M14.5 8.5h.01M10 11.5a2.5 2.5 0 0 0 4 0" /><path d="M6.5 14.5 5 21M17.5 14.5 19 21M8 21h8" /></svg>),
  cross: (p) => (<svg {...base} {...p}><path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6z" /></svg>),
  eye: (p) => (<svg {...base} {...p}><path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z" /><circle cx="12" cy="12" r="2.5" /></svg>),
  pill: (p) => (<svg {...base} {...p}><rect x="3.5" y="8.5" width="17" height="7" rx="3.5" transform="rotate(-45 12 12)" /><path d="m8.5 8.5 7 7" /></svg>),
};

export default Icon;