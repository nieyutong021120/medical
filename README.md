# 华佗医疗智能助手

基于 **Huatuo-Lite 开源医疗问答库（177,703 条真实问答）** 的 RAG 医疗问答系统：Next.js 前端 + FastAPI 后端 + FAISS/BM25 混合检索 + Moonshot LLM 生成。回答附参考问答溯源、建议科室与相关疾病提示。

> 演示项目：回答仅供健康参考，不能替代执业医师的面诊、诊断与治疗建议。如有不适请及时到正规医疗机构就诊。

## 功能特性

- **智能医疗问答**：SSE 真流式输出；先检索后作答，回答基于知识库真实医生问答生成
- **质量分层检索**：FAISS 向量（bge-small-zh-v1.5，仅嵌入问题）+ jieba-BM25（问题+答案）→ RRF 融合 → 按数据集 score(4/5) 加权
- **智能导诊**：对 top-k 检索结果按 rrf 分数加权投票，给出建议科室（16 个科室标签）
- **相关疾病提示**：聚合检索结果的 related_diseases 字段，去重取前 6
- **相似病例展示**：每条回答附参考 QA 卡片，点击打开抽屉查看医生原始回答
- **知识库概览**：总量/科室分布/平均回答长度统计（来自后端实测），24 条 score=5 样例可搜索、按科室筛选
- **会话管理**：后端 SQLite 持久化 + httponly Cookie 身份隔离；多会话、删除可撤销、消息反馈

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | Next.js 15（App Router，`output: 'export'` 静态导出）、React 19、Tailwind CSS v4、anime.js 4 |
| 后端 | FastAPI + uvicorn，SSE 流式接口，SQLite（WAL）会话存储 |
| RAG | LangChain 模块骨架：ingestion / indexing / retrieval / dialogue / generation |
| 检索 | FAISS（向量）+ BM25（jieba 分词），RRF 融合 + score 加权 |
| 模型 | 嵌入 BAAI/bge-small-zh-v1.5（本地 HF 缓存，离线加载）；生成 kimi-k3（Moonshot API，temperature 固定 1） |
| 数据 | `Huatuo-Lite/format_data.jsonl`（question / answer / score / label / related_diseases） |

## 快速开始

> **仓库不含大体积文件**：检索索引 `medical_rag/vector_index/`（约 1.2GB，可本地重建）与数据集 `Huatuo-Lite/format_data.jsonl`（约 132MB，开源可下载）均超过 GitHub 单文件 100MB 上限，已在 `.gitignore` 中排除。克隆后需按下面步骤自行准备数据与索引。

### 1. 准备数据集与 Python 依赖

从 HuggingFace 下载 [FreedomIntelligence/Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite) 的 `format_data.jsonl`，放到 `Huatuo-Lite/format_data.jsonl`（仓库已保留该目录的 `README.md` 与 `my_dataset.py`）。国内网络可用镜像 `hf-mirror.com`。

```bash
# Python 依赖（建议 conda/venv 独立环境）
pip install -r medical_rag/requirements.txt
```

### 2. 离线构建索引（首次必须，CPU 嵌入 17.8 万条约需数十分钟）

```bash
python medical_rag/scripts/build_index.py        # 生成 FAISS index.faiss / index.pkl + bm25.pkl
python medical_rag/scripts/build_inverted_bm25.py  # 由 bm25.pkl 转换 inverted.pkl（逐位一致性校验）
```

产物在 `medical_rag/vector_index/`。之后启动直接加载，不再重建。（示例命令中的 `python` 替换为你的环境解释器，如 `D:/miniconda/envs/all-in-rag/python.exe`。）

### 3. 启动后端（仓库根目录）

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

- 先复制 `medical_rag/.env.example` 为 `medical_rag/.env`，填入 `MOONSHOT_API_KEY`（override=True 为唯一权威来源）。
- 前端地址从 `backend/.env` 的 `FRONTEND_URL` 读取（用作 CORS 跨域白名单，默认 `http://localhost:3000`）。模板见 `backend/.env.example`。监听 host/port 由 uvicorn 命令行控制。

### 4. 启动前端

```bash
npm install
npm run dev
```

打开 <http://localhost:3000>。前端 API 基址由根 `.env` 的 `NEXT_PUBLIC_API_URL` 配置（默认 `http://localhost:8000`，模板见 `.env.example`）；后端未启动时展示样例数据并给出错误气泡。

> **部署注意**：前端为静态导出（`output: 'export'`），`NEXT_PUBLIC_API_URL` 在 **构建期** 注入，改地址后须重新 `npm run build`。该地址须与访问前端所用 host **同站**，否则 SameSite=Lax 的 client_id Cookie 不生效；同时把前端地址设为后端 `backend/.env` 的 `FRONTEND_URL`。

### 生产构建

```bash
# 先在根 .env 设好 NEXT_PUBLIC_API_URL（指向部署后的后端地址），再构建
npm run build      # 静态产物输出到 out/，用任意静态服务器托管
```

注意：不要在 dev 服务器运行时执行 build——两者共享 `.next` 缓存会冲突。

## API 一览

| 接口 | 说明 |
| --- | --- |
| `GET /api/health` | 就绪状态 + 知识库统计（总量/科室分布/平均分等） |
| `POST /api/ask` | 一次性问答 `{question}` → `{answer, sources, department, related_diseases, route, disclaimer}` |
| `POST /api/ask/stream` | SSE 流式：`meta`（溯源/科室/疾病/免责声明）→ `delta`（文本增量）→ `done` / `error` |
| `/api/conversations*`、`/api/messages*` | 会话与消息 CRUD（Cookie client_id 隔离，id 由后端生成） |

## 项目结构

```
app/                        Next.js 外壳：layout / page / globals.css（唯一 CSS 入口）
components/medical/         前端组件：MedicalAssistant 主外壳、ChatPanel、ChatMessage（流式）、
                            KnowledgePanel、SourceDrawer、Sidebar、Icons
data/medicalKnowledge.js    内容层：助手信息、6 科室卡片、热门问题、24 条 score=5 样例、兜底统计
lib/config.js               前端 API 基址（NEXT_PUBLIC_API_URL，构建期注入）
lib/medicalApi.js           后端客户端：health / ask / SSE 流式解析
lib/medicalStore.js         会话 CRUD 客户端
.env / .env.example         前端配置（NEXT_PUBLIC_API_URL）
backend/main.py             FastAPI：包装 medical_rag，SSE 与会话接口
backend/db.py               SQLite 会话存储（data/medical_assistant.db）
backend/.env / .env.example 后端配置（FRONTEND_URL，CORS 白名单）
medical_rag/                RAG 引擎
  config.py                 MedicalConfig（数据路径/模型/top_k 等）
  main.py                   MedicalSystem：检索→科室投票→疾病聚合→生成（含流式）
  kb_modules/               ingestion / indexing / retrieval / dialogue / generation
  scripts/build_index.py    离线索引构建器
  scripts/build_inverted_bm25.py  BM25 倒排索引转换 + 逐位一致性校验
  scripts/eval_retrieval.py 检索离线评测（结果 eval_results.json）
  vector_index/             FAISS + BM25 持久化产物（本地重建，不入库）
Huatuo-Lite/                数据集目录（README.md / my_dataset.py 入库；
                            format_data.jsonl 约 132MB 需自行下载，不入库）
```

## 检索与生成机制

1. **双路召回**：向量路只嵌入问题（question-to-question 语义匹配）；BM25 路索引问题+答案（药名/症状词可能只出现在答案里），采用倒排索引 InvertedBM25Retriever（numpy 向量化打分，分数与原 rank_bm25 逐位一致，实测 ~20ms/查询，较前向扫描提速约 64×）。各取 top-10。
2. **RRF 融合**：按 chunk_id 去重，`rrf = Σ 1/(60+rank)`。
3. **质量加权**：`final = rrf * (1 + 0.05 * (score - 4))`，score=5 的医生优质回答获得约 5% 提升。
4. **导诊与疾病**：top-5 结果按 final 分数对 label 加权投票得建议科室；related_diseases 聚合去重取前 6——零额外 LLM 调用。
5. **生成约束**：提示词禁止给出确定性诊断与处方剂量；急症症状优先提示拨打 120；上下文最多 3000 字符，每条参考带【参考i】科室|评分标记。

## 已知限制

- 数据集为历史开源问答，非权威临床指南；score=4 占 87%，存在少量噪声回答
- 答案偏短（平均百余字），不适合深度科普；急诊科样本仅 1 条，不做急症分诊
- Moonshot 组织级限速 RPM=3，连续提问可能触发重试等待

## 声明

本项目为演示用途，所有回答不构成医疗建议；具体诊疗请以正规医疗机构执业医师意见为准。