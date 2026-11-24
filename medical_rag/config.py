"""
医疗问答RAG系统配置文件（改自 hr_compliance_rag/config.py，模型配置沿用）
"""

from dataclasses import asdict, dataclass
from typing import Dict, Any


@dataclass
class MedicalConfig:
    """医疗智能助手RAG系统配置类"""

    # 路径配置
    # 旧：raw_data_path/laws_md_path 双路径（法规JSON+Markdown），QA对单一jsonl即可
    data_path: str = "../Huatuo-Lite/format_data.jsonl"  # 华佗QA对数据
    index_save_path: str = "./vector_index"              # 向量索引持久化目录

    # 模型配置（沿用现有项目）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k3"

    # 检索配置
    top_k: int = 5          # 最终返回的参考QA数量
    retriever_k: int = 10   # 单路(向量/BM25)召回候选数，供RRF融合

    # 生成配置（沿用现有项目）
    temperature: float = 1
    max_tokens: int = 2048

    # Agent 配置（阶段二：复杂/模糊问题走工具调用 Agent，简单问题保持快速路径）
    agent_enabled: bool = True       # 关闭后 complex_or_ambiguous 也走快速路径
    agent_max_iterations: int = 2    # Agent 最多决策轮数（每轮 1 次 LLM 调用，控制 RPM 消耗）

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MedicalConfig':
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = MedicalConfig()