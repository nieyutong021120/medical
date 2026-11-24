"""
医疗智能助手RAG系统 - 主程序（改自 hr_compliance_rag/main.py）

流程：意图路由(三分类) -> [明确问题] 查询重写 -> 混合检索(评分加权) -> 生成
                    -> [复杂/模糊] Agent 工具循环(检索/澄清追问) -> 生成或追问
导诊科室与相关疾病为检索副产品：top-k 的 label 按 rrf_score 加权投票、
related_diseases 聚合去重，零额外模型调用。
"""

import os
import sys
import logging
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, MedicalConfig
from kb_modules import (
    IngestionModule,
    IndexingModule,
    RetrievalModule,
    DialogueModule,
    GenerationModule,
    AgentModule,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OUT_OF_SCOPE_ANSWER = (
    "这个问题超出了医疗健康助手的服务范围。我可以解答症状咨询、疾病知识、"
    "用药常识、就医科室推荐等健康问题，试着问我一个健康相关的问题吧。"
)
NO_MATCH_ANSWER = "抱歉，没有检索到相关的参考问答。请尝试换一种描述，或补充症状细节。"


class MedicalSystem:
    """医疗智能助手系统"""

    def __init__(self, config: MedicalConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.ingestion: IngestionModule = None
        self.indexing: IndexingModule = None
        self.retrieval: RetrievalModule = None
        self.dialogue: DialogueModule = None
        self.generation: GenerationModule = None
        self.agent: AgentModule = None

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

    def initialize_system(self):
        print("正在初始化医疗RAG系统...")
        self.ingestion = IngestionModule(self.config.data_path)
        self.indexing = IndexingModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
        )
        self.dialogue = DialogueModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
        )
        self.generation = GenerationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        print("系统初始化完成！")

    def build_knowledge_base(self):
        print("\n正在构建知识库...")
        self.ingestion.load_documents()

        vectorstore = self.indexing.load_index()
        if vectorstore is None:
            print("未找到已保存的向量索引，开始构建新索引（17.8万条，耗时较长）...")
            vectorstore = self.indexing.build_vector_index(self.ingestion.documents)
            self.indexing.save_index()

        bm25 = self.indexing.load_bm25_index()
        if bm25 is None:
            print("未找到BM25缓存，开始构建...")
            bm25 = self.indexing.build_bm25_index(
                self.ingestion.get_bm25_documents(), k=self.config.retriever_k
            )
            self.indexing.save_index()

        self.retrieval = RetrievalModule(vectorstore, bm25, retriever_k=self.config.retriever_k)

        # 阶段二：复杂/模糊问题走工具调用 Agent（agent_enabled=False 时全部走快速路径）
        if self.config.agent_enabled:
            self.agent = AgentModule(
                self.retrieval,
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_iterations=self.config.agent_max_iterations,
                top_k=self.config.top_k,
            )

        stats = self.ingestion.get_statistics()
        print("\n知识库统计:")
        print(f"   QA对总数: {stats['total_qa']}")
        print(f"   科室数: {len(stats['departments'])}")
        print(f"   平均答案长度: {stats['avg_answer_len']} 字")
        print("知识库构建完成！")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _vote_department(chunks) -> str:
        """top-k 科室按 rrf_score 加权投票"""
        votes = defaultdict(float)
        for c in chunks:
            votes[c.metadata.get("label", "其他")] += c.metadata.get("rrf_score", 0.0)
        return max(votes, key=votes.get) if votes else ""

    @staticmethod
    def _collect_diseases(chunks, limit: int = 6):
        seen, out = set(), []
        for c in chunks:
            dis = (c.metadata.get("related_diseases") or "").strip()
            if dis and dis not in seen:
                seen.add(dis)
                out.append(dis)
            if len(out) >= limit:
                break
        return out

    def _retrieve(self, question: str, history=None):
        """路由+重写+检索，返回 (route, chunks, rewritten)；越界/无命中时 chunks 为空
        （阶段二起主流程改用 _route_and_prepare，此方法保留供脚本/交互模式参考）"""
        route = self.dialogue.query_router(question)
        if route == 'out_of_scope':
            return route, [], question
        # 旧：rewritten = self.dialogue.query_rewrite(question)（单轮重写）
        rewritten = (self.dialogue.rewrite_with_history(question, history)
                     if history else self.dialogue.query_rewrite(question))
        chunks = self.retrieval.hybrid_search(rewritten, top_k=self.config.top_k)
        return route, chunks, rewritten

    def _rewrite_and_search(self, question: str, history=None):
        """快速路径的重写+检索（不含路由），也用作 Agent 回退"""
        rewritten = (self.dialogue.rewrite_with_history(question, history)
                     if history else self.dialogue.query_rewrite(question))
        chunks = self.retrieval.hybrid_search(rewritten, top_k=self.config.top_k)
        return chunks, rewritten

    def _route_and_prepare(self, question: str, history=None):
        """
        阶段二分流（生成器）：
          途中 yield {"_event":"tool",…}（Agent 工具轨迹，供 SSE 展示）
          末尾 yield {"_event":"prep", route, chunks, rewritten, clarification}
        路由三分：medical_qa→快速路径；complex_or_ambiguous→Agent（澄清/检索作答/回退）；
        out_of_scope→兜底。调用次数预算：快速=3，Agent一次决策=3，澄清=2，最坏=4。
        """
        route = self.dialogue.query_router(question, history)
        if route == 'out_of_scope':
            yield {"_event": "prep", "route": route, "chunks": [],
                   "rewritten": question, "clarification": ""}
            return

        if route == 'complex_or_ambiguous' and self.agent is not None:
            result = None
            for ev in self.agent.run(question, history):
                if ev.get("_event") == "agent_result":
                    result = ev
                else:
                    yield ev  # tool 事件原样透传
            if result["action"] == "clarify":
                yield {"_event": "prep", "route": "clarify", "chunks": [],
                       "rewritten": question, "clarification": result["clarification"]}
                return
            if result["action"] == "answer":
                yield {"_event": "prep", "route": "agent", "chunks": result["chunks"],
                       "rewritten": result["rewritten"], "clarification": ""}
                return
            # fallback：Agent 决策失败/检索为空 → 回退快速路径
            logger.info("Agent 回退快速路径")

        chunks, rewritten = self._rewrite_and_search(question, history)
        yield {"_event": "prep", "route": "medical_qa", "chunks": chunks,
               "rewritten": rewritten, "clarification": ""}

    def ask_with_sources(self, question: str, history=None) -> dict:
        """后端 API 用：返回 {route, answer, sources, department, related_diseases, rewritten_query}"""
        if not all([self.retrieval, self.generation, self.dialogue]):
            raise ValueError("请先构建知识库")

        # 旧：route, chunks, rewritten = self._retrieve(question, history)（无 Agent 分流）
        prep = None
        for ev in self._route_and_prepare(question, history):
            if ev.get("_event") == "prep":
                prep = ev  # 非流式接口丢弃 tool 轨迹事件
        route, chunks, rewritten = prep["route"], prep["chunks"], prep["rewritten"]
        if route == 'out_of_scope':
            return {"route": route, "answer": OUT_OF_SCOPE_ANSWER, "sources": [],
                    "department": "", "related_diseases": [], "rewritten_query": question}
        if route == 'clarify':
            return {"route": route, "answer": prep["clarification"], "sources": [],
                    "department": "", "related_diseases": [], "rewritten_query": question}
        if not chunks:
            return {"route": "no_match", "answer": NO_MATCH_ANSWER, "sources": [],
                    "department": "", "related_diseases": [], "rewritten_query": rewritten}

        # 旧：answer = self.generation.generate_answer(question, chunks)（单轮）
        answer = self.generation.generate_answer(question, chunks, history)
        sources = [{
            "id": c.metadata.get("chunk_id", ""),
            "question": c.page_content,
            "answer": c.metadata.get("answer", ""),
            "label": c.metadata.get("label", ""),
            "score": c.metadata.get("score", 0),
            "related_diseases": c.metadata.get("related_diseases", ""),
        } for c in chunks]
        return {
            "route": route,
            "answer": answer,
            "sources": sources,
            "department": self._vote_department(chunks),
            "related_diseases": self._collect_diseases(chunks),
            "rewritten_query": rewritten,
        }

    def ask_stream(self, question: str, history=None):
        """SSE 用：途中可 yield {"_event":"tool",…}，随后 yield meta(dict)，再逐段 yield 回答文本(str)"""
        if not all([self.retrieval, self.generation, self.dialogue]):
            raise ValueError("请先构建知识库")

        # 旧：route, chunks, rewritten = self._retrieve(question, history)（无 Agent 分流）
        prep = None
        for ev in self._route_and_prepare(question, history):
            if ev.get("_event") == "prep":
                prep = ev
            else:
                yield ev  # tool 事件透传给后端 SSE
        route, chunks, rewritten = prep["route"], prep["chunks"], prep["rewritten"]
        if route == 'out_of_scope':
            yield {"route": route, "sources": [], "department": "", "related_diseases": [],
                   "rewritten_query": question}
            yield OUT_OF_SCOPE_ANSWER
            return
        if route == 'clarify':
            yield {"route": route, "sources": [], "department": "", "related_diseases": [],
                   "rewritten_query": question}
            yield prep["clarification"]
            return
        if not chunks:
            yield {"route": "no_match", "sources": [], "department": "", "related_diseases": [],
                   "rewritten_query": rewritten}
            yield NO_MATCH_ANSWER
            return

        sources = [{
            "id": c.metadata.get("chunk_id", ""),
            "question": c.page_content,
            "answer": c.metadata.get("answer", ""),
            "label": c.metadata.get("label", ""),
            "score": c.metadata.get("score", 0),
            "related_diseases": c.metadata.get("related_diseases", ""),
        } for c in chunks]
        yield {
            "route": route,
            "sources": sources,
            "department": self._vote_department(chunks),
            "related_diseases": self._collect_diseases(chunks),
            "rewritten_query": rewritten,
        }
        # 旧：yield from self.generation.generate_answer_stream(question, chunks)（单轮）
        yield from self.generation.generate_answer_stream(question, chunks, history)

    # ------------------------------------------------------------------ #
    def run_interactive(self):
        print("=" * 60)
        print("医疗智能助手（Huatuo-Lite RAG）")
        print("=" * 60)
        self.initialize_system()
        self.build_knowledge_base()
        print("交互式问答 (输入'退出'结束):")
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                result = self.ask_with_sources(user_input)
                print(f"\n[意图: {result['route']} | 建议科室: {result['department']}]")
                print(result["answer"])
                if result["related_diseases"]:
                    print("相关疾病:", "、".join(result["related_diseases"]))
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")


def main():
    try:
        MedicalSystem().run_interactive()
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()