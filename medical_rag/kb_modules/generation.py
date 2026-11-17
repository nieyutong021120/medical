"""
生成集成模块 - 医疗问答生成（改自 hr_compliance_rag/kb_modules/generation.py）

旧：法条上下文+角色视角+回答末尾拼接依据条款。现改为：
- 上下文为检索到的参考QA对（患者问题+医生回答+科室+评分）
- 溯源不再拼进answer文本，由后端以结构化sources返回，前端渲染卡片
- 免责声明仍由代码确定性提供（DISCLAIMER常量），不依赖LLM自觉
- 安全约束：不做确定性诊断、不给处方剂量、建议就医
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

DISCLAIMER = "\n\n---\n⚠️ 本回答基于华佗开源医疗问答库（Huatuo-Lite）检索生成，仅供健康参考，不能替代执业医师的面诊、诊断与治疗建议。如有不适请及时到正规医疗机构就诊。"


class GenerationModule:
    """生成集成模块 - 医疗问答"""

    def __init__(self, model_name: str = "kimi-k3", temperature: float = 0.2, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        logger.info(f"正在初始化生成LLM: {self.model_name}")
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=5,
            moonshot_api_key=api_key,
        )
        logger.info("生成LLM初始化完成")

    # ------------------------------------------------------------------ #
    def _build_context(self, chunks: List[Document], max_length: int = 3000) -> str:
        if not chunks:
            return "暂无相关参考问答。"
        parts, cur = [], 0
        for i, doc in enumerate(chunks, 1):
            m = doc.metadata
            header = f"【参考{i}】科室: {m.get('label', '')} | 质量评分: {m.get('score', '')}"
            text = f"{header}\n患者问题: {doc.page_content.strip()}\n医生回答: {m.get('answer', '').strip()}\n"
            if cur + len(text) > max_length:
                break
            parts.append(text)
            cur += len(text)
        divider = "\n" + "=" * 50 + "\n"
        return divider + divider.join(parts)

    # ------------------------------------------------------------------ #
    def _answer_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
你是一位专业、严谨、有同理心的医疗健康助手。请严格依据下方检索到的参考问答回答用户问题。

用户问题: {question}

检索到的参考问答:
{context}

要求:
- 综合参考问答给出通俗、有条理的回答，可分点说明
- 不得编造参考问答之外的医学结论；参考未覆盖的部分如实说明"现有参考资料未涉及"
- 不做确定性诊断（如"你患的就是XX病"），不给出处方药的具体剂量用法
- 涉及急重症状（如剧烈胸痛、呼吸困难、大量出血、意识障碍）时，优先提醒立即就医或拨打120
- 回答结尾用一句话建议必要时到相应科室就诊
- 不要提及"参考1/参考2"等编号，直接组织成自然回答

回答:""")

    def _chain(self, context: str):
        return (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | self._answer_prompt() | self.llm | StrOutputParser()
        )

    def generate_answer(self, query: str, chunks: List[Document]) -> str:
        # 旧：answer + _build_citations(chunks) + DISCLAIMER（溯源改为结构化sources由后端返回）
        return self._chain(self._build_context(chunks)).invoke(query)

    def generate_answer_stream(self, query: str, chunks: List[Document]):
        for chunk in self._chain(self._build_context(chunks)).stream(query):
            yield chunk