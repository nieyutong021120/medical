"""
医疗智能助手RAG系统 - 主程序（改自 hr_compliance_rag/main.py）

流程：意图路由 -> 查询重写 -> 混合检索(评分加权) -> 生成
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

    def _retrieve(self, question: str):
        """路由+重写+检索，返回 (route, chunks)；越界/无命中时 chunks 为空"""
        route = self.dialogue.query_router(question)
        if route == 'out_of_scope':
            return route, []
        rewritten = self.dialogue.query_rewrite(question)
        chunks = self.retrieval.hybrid_search(rewritten, top_k=self.config.top_k)
        return route, chunks

    def ask_with_sources(self, question: str) -> dict:
        """后端 API 用：返回 {route, answer, sources, department, related_diseases}"""
        if not all([self.retrieval, self.generation, self.dialogue]):
            raise ValueError("请先构建知识库")

        route, chunks = self._retrieve(question)
        if route == 'out_of_scope':
            return {"route": route, "answer": OUT_OF_SCOPE_ANSWER, "sources": [],
                    "department": "", "related_diseases": []}
        if not chunks:
            return {"route": "no_match", "answer": NO_MATCH_ANSWER, "sources": [],
                    "department": "", "related_diseases": []}

        answer = self.generation.generate_answer(question, chunks)
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
        }

    def ask_stream(self, question: str):
        """SSE 用：先 yield meta(dict)，再逐段 yield 回答文本(str)"""
        if not all([self.retrieval, self.generation, self.dialogue]):
            raise ValueError("请先构建知识库")

        route, chunks = self._retrieve(question)
        if route == 'out_of_scope':
            yield {"route": route, "sources": [], "department": "", "related_diseases": []}
            yield OUT_OF_SCOPE_ANSWER
            return
        if not chunks:
            yield {"route": "no_match", "sources": [], "department": "", "related_diseases": []}
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
        }
        yield from self.generation.generate_answer_stream(question, chunks)

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