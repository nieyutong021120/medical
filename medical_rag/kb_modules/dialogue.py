"""
对话理解模块 - 意图路由 + 查询重写 + 越界兜底
（改自 hr_compliance_rag/kb_modules/dialogue.py，场景从劳动用工改为医疗健康）

旧：四分类(list/detail/general/out_of_scope)。QA对形态下清单类无意义，
简化为 medical_qa / out_of_scope 两类；重写规则改为"口语症状→规范医学表述"。
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class DialogueModule:
    """对话理解模块 - 意图路由与查询重写"""

    def __init__(self, model_name: str = "kimi-k3", temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        logger.info(f"正在初始化对话LLM: {self.model_name}")
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=512,
            max_retries=5,
            moonshot_api_key=api_key,
        )
        logger.info("对话LLM初始化完成")

    # ------------------------------------------------------------------ #
    def query_router(self, query: str) -> str:
        """
        意图路由：
          medical_qa   - 症状/疾病/用药/就医/健康科普等医疗健康问题
          out_of_scope - 与医疗健康无关（闲聊、天气、编程等），兜底防幻觉
        """
        prompt = ChatPromptTemplate.from_template("""
你是医疗健康助手的意图分类器。请将用户问题归为以下两类之一：

1. 'medical_qa' - 医疗健康相关：症状咨询、疾病知识、用药常识、检查解读、就医科室、预防保健、康复护理等。例：最近总是头晕是怎么回事？高血压能吃党参吗？孩子发烧38度怎么办？
2. 'out_of_scope' - 与医疗健康无关：闲聊、天气、写诗、编程、法律、财务等。例：今天天气如何？帮我写一首诗？

请只返回一个词：medical_qa 或 out_of_scope

用户问题: {query}

分类结果:""")
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        result = chain.invoke(query).strip().lower()
        if result in ['medical_qa', 'out_of_scope']:
            return result
        return 'medical_qa'

    # ------------------------------------------------------------------ #
    @staticmethod
    def _clean_rewrite(response: str, query: str) -> str:
        """清洗重写输出：模型偶发附带"**判断说明**：…"等解释文字，会污染检索与展示。"""
        text = response.strip().split("\n")[0].strip()
        for marker in ("判断说明", "说明：", "解释：", "理由："):
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx].strip()
        text = text.strip("*").strip()
        # 空/过短/异常冗长（重写不应远超原问）一律回退原查询
        if not text or len(text) < 2 or len(text) > max(100, len(query) * 3):
            logger.warning(f"重写输出异常('{response[:80]}…')，回退原查询: '{query}'")
            return query
        return text

    # ------------------------------------------------------------------ #
    def query_rewrite(self, query: str) -> str:
        """口语化症状描述重写为规范医学表述，提升与库中患者提问的匹配度"""
        prompt = PromptTemplate(
            template="""
你是医疗健康领域的查询分析助手。判断是否需要重写用户查询以提升医疗问答检索效果。

原始查询: {query}

规则：
1. 具体明确(直接返回原查询)：含明确疾病/症状/药品名，如"高血压能吃党参吗""糖尿病的诊断标准""痔疮手术后注意事项"
2. 模糊口语(需重写为规范医学表述)：如"肚子一阵一阵拧着疼"→"阵发性腹痛的原因"，"上不来气"→"呼吸困难 胸闷"，"浑身没劲老想睡觉"→"乏力 嗜睡 原因"

重写原则：保持原意、使用规范医学词汇、简洁，不要添加用户未提及的症状。

请输出最终查询(无需重写就返回原查询):""",
            input_variables=["query"]
        )
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        response = chain.invoke(query).strip()
        # 旧：仅防空串；现统一清洗（模型偶发附带"判断说明"等解释文字，污染检索与展示）
        # if not response or len(response) < 2:
        #     logger.warning(f"查询重写返回空/异常('{response}')，回退原查询: '{query}'")
        #     return query
        response = self._clean_rewrite(response, query)
        if response != query:
            logger.info(f"查询已重写: '{query}' -> '{response}'")
        return response

    # ------------------------------------------------------------------ #
    def rewrite_with_history(self, query: str, history: List[dict]) -> str:
        """多轮指代消解改写：结合历史把当前问题补全为自包含查询（替代单轮 query_rewrite，不额外增加调用）"""
        # 旧：[二期预留] 空实现 return query
        lines = [
            ("用户" if m["role"] == "user" else "助手") + ": " + m["text"]
            for m in history
        ]
        prompt = PromptTemplate(
            template="""
你是医疗健康领域的查询分析助手。请结合对话历史，把用户当前问题改写为独立、完整、适合检索的查询。

对话历史:
{history}

当前问题: {query}

规则：
1. 当前问题含指代（它/这个病/那种药等）或省略主语时，依据历史补全。如历史在谈糖尿病，"它有什么并发症？"→"糖尿病有什么并发症？"
2. 当前问题已自包含时直接返回原问题；不要添加用户未提及的症状或疾病
3. 使用规范医学词汇、保持简洁

请输出最终查询:""",
            input_variables=["history", "query"]
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"history": "\n".join(lines), "query": query}).strip()
        # 与 query_rewrite 相同的防御：清洗 + 空/异常回退原查询
        response = self._clean_rewrite(response, query)
        if response != query:
            logger.info(f"查询已重写(带历史): '{query}' -> '{response}'")
        return response