"""
Agent 模块 - 复杂/模糊问题的工具调用循环（阶段二新增）

设计要点（旧：受 Moonshot RPM=3 约束；现换本地 OpenAI 兼容端点，限速放宽，轮数预算保留）：
- 混合路由：仅 complex_or_ambiguous 问题进入 Agent；明确问题保持快速路径。
- 工具：search_medical_qa（包装混合检索，返回压缩QA摘要）+ ask_clarification（终止型追问）。
- 最多 max_iterations 轮决策（每轮 1 次 LLM 调用）；医疗回答永远交给生成模块基于检索
  结果产出，Agent 自身不自由生成医学内容（防幻觉）。
- 无效参数 / 检索为空 / 模型不调用工具 → 回退快速路径；轮数用尽但有检索结果 → 强制生成。
- LLM 用 ChatOpenAI 指向 OpenAI 兼容端点（API_BASE_URL，默认 https://api.llm.ustc.edu.cn/v1；
  旧：指向 Moonshot 端点，因 MoonshotChat.bind_tools 是基类默认实现不可用）。
- run() 是生成器：途中 yield {"_event":"tool",...} 供 SSE 展示工具轨迹，
  最后 yield {"_event":"agent_result", action: answer|clarify|fallback, ...}。
"""

import os
import logging
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 检索结果喂回 Agent 时的答案截断长度（控制 token 消耗）
ANSWER_SNIPPET = 200

SYSTEM_PROMPT = """你是医疗健康助手的决策核心，负责处理复杂或信息不足的用户问题。你有两个工具：
1. search_medical_qa：检索医疗问答知识库。凡需要给出医疗回答，必须先检索，严禁凭自身知识直接回答医学问题。
2. ask_clarification：当用户问题缺少关键信息（如症状部位、持续时间、诱因、伴随症状），以致无法有效检索时，向用户提出一个最关键的澄清问题。调用后本轮立即结束。

决策规则：
- 结合对话历史理解当前问题；传给 search_medical_qa 的 query 必须自包含、使用规范医学词汇。
- 信息明显不足时优先 ask_clarification：只问一个问题，聚焦最有助于缩小方向的关键信息，语气友好。
- 检索结果返回后，若足以支撑回答，就不要再调用任何工具（系统会基于检索结果生成最终回答）；仅当结果明显不相关时才换关键词再检索一次。
- 你最多有 {max_iterations} 轮决策机会，请节约使用。"""


class AgentModule:
    """工具调用 Agent：检索 / 澄清追问，产出 chunks 或澄清问题"""

    # 旧：def __init__(self, retrieval, model_name: str = "kimi-k3", temperature: float = 1,
    def __init__(self, retrieval, model_name: str = "glm-5.3-flash", temperature: float = 0.2,
                 max_iterations: int = 2, top_k: int = 5):
        self.retrieval = retrieval
        self.max_iterations = max_iterations
        self.top_k = top_k
        # 旧：Moonshot 官方端点 + MOONSHOT_API_KEY；现换本地 OpenAI 兼容端点，
        # API_KEY / API_BASE_URL 来自 medical_rag/.env（后端 config.py 以 override=True 加载）
        # api_key = os.getenv("MOONSHOT_API_KEY")
        # if not api_key:
        #     raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("请设置 API_KEY 环境变量")
        logger.info(f"正在初始化Agent LLM: {model_name} (max_iterations={max_iterations})")
        # 旧：self.llm = ChatOpenAI(
        # 旧：    model=model_name,
        # 旧：    base_url="https://api.moonshot.cn/v1",
        # 旧：    api_key=api_key,
        # 旧：    temperature=temperature,   # kimi-k3 仅接受 1
        # 旧：    max_tokens=1024,
        # 旧：    max_retries=5,
        # 旧：)
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=os.getenv("API_BASE_URL", "https://api.llm.ustc.edu.cn/v1"),
            api_key=api_key,
            temperature=temperature,   # 旧：kimi-k3 仅接受 1；GLM 支持 0~1
            max_tokens=1024,
            max_retries=5,
        )
        logger.info("Agent LLM初始化完成")

    # ------------------------------------------------------------------ #
    def _build_tools(self, state):
        """每次 run 构建工具：检索/澄清结果写入本次调用私有的 state，避免并发请求串扰"""
        retrieval = self.retrieval
        default_k = self.top_k

        @tool
        def search_medical_qa(query: str, top_k: int = 5) -> str:
            """检索医疗问答知识库，返回最相关的患者提问与医生回答摘要。query 应为自包含的规范医学检索短语（如"阵发性腹痛 原因"）；top_k 为返回条数，默认5。"""
            q = (query or "").strip()
            if not q:
                return "错误：query 不能为空"
            try:
                k = max(1, min(int(top_k or default_k), 10))
            except (TypeError, ValueError):
                k = default_k
            chunks = retrieval.hybrid_search(q, top_k=k)
            state["chunks"] = chunks
            state["rewritten"] = q
            if not chunks:
                return "未检索到相关 QA，请更换关键词重试或改用 ask_clarification。"
            return AgentModule._summarize(chunks)

        @tool
        def ask_clarification(question: str) -> str:
            """当用户问题缺少关键信息、无法有效检索时，向用户提出一个澄清问题。question 为展示给用户的追问文本；调用后本轮立即结束，等待用户补充。"""
            q = (question or "").strip()
            if not q:
                return "错误：question 不能为空"
            state["clarification"] = q
            return "澄清问题已发送给用户，等待用户回复。"

        return [search_medical_qa, ask_clarification]

    @staticmethod
    def _summarize(chunks) -> str:
        """压缩检索结果为文本摘要（问题+答案截断+科室+评分+ID），控制回传 token"""
        lines = []
        for i, c in enumerate(chunks):
            answer = (c.metadata.get("answer", "") or "")[:ANSWER_SNIPPET]
            lines.append(
                f"[{i + 1}] 问题: {c.page_content}\n"
                f"回答(节选): {answer}\n"
                f"科室: {c.metadata.get('label', '')} | 质量分: {c.metadata.get('score', 0)} | "
                f"ID: {c.metadata.get('chunk_id', '')}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ #
    def run(self, question: str, history: List[dict] = None):
        """
        生成器：
          途中 yield {"_event":"tool","name":…,"args":…}
          末尾 yield {"_event":"agent_result","action":"answer"|"clarify"|"fallback",
                      "chunks":[…],"rewritten":…,"clarification":…}
        action=answer   → chunks 交生成模块作答
        action=clarify  → clarification 直接作为回复（不检索不生成）
        action=fallback → 回退快速路径（重写+检索+生成）
        """
        state = {"chunks": [], "rewritten": "", "clarification": ""}
        tools = self._build_tools(state)
        llm_with_tools = self.llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        user_content = question
        if history:
            lines = [
                ("用户" if m["role"] == "user" else "助手") + ": " + m["text"]
                for m in history
            ]
            user_content = "对话历史:\n" + "\n".join(lines) + "\n\n当前问题: " + question
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(max_iterations=self.max_iterations)),
            HumanMessage(content=user_content),
        ]

        def result(action):
            return {
                "_event": "agent_result", "action": action,
                "chunks": state["chunks"],
                "rewritten": state["rewritten"] or question,
                "clarification": state["clarification"],
            }

        for it in range(self.max_iterations):
            logger.info(f"Agent 决策轮 {it + 1}/{self.max_iterations}: '{question[:50]}'")
            ai = llm_with_tools.invoke(messages)
            messages.append(ai)
            calls = ai.tool_calls or []
            if not calls:
                # 模型不再调用工具：有检索结果则收尾作答，否则视为决策失败回退
                break
            for tc in calls:
                name, args = tc["name"], (tc.get("args") or {})
                yield {"_event": "tool", "name": name, "args": args}
                if name not in tool_map:
                    logger.warning(f"Agent 调用未知工具 '{name}'，回退快速路径")
                    yield result("fallback")
                    return
                try:
                    observation = tool_map[name].invoke(args)
                except Exception as e:
                    logger.warning(f"Agent 工具 '{name}' 参数无效({e})，回退快速路径")
                    yield result("fallback")
                    return
                if name == "ask_clarification":
                    if not state["clarification"]:
                        yield result("fallback")
                        return
                    yield result("clarify")
                    return
                # search_medical_qa：空结果 → 回退快速路径（重写后再检索一次）
                if not state["chunks"]:
                    logger.info("Agent 检索为空，回退快速路径")
                    yield result("fallback")
                    return
                messages.append(ToolMessage(content=observation, tool_call_id=tc["id"]))
                # 澄清工具若与检索同批出现，检索结果已记录，继续下一轮由模型决定

        if state["chunks"]:
            # 含"轮数用尽仍有检索结果 → 强制生成"的情形
            yield result("answer")
        else:
            yield result("fallback")