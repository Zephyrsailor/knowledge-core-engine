# CLAUDE.md - KnowledgeCore Engine 项目指导文档

## TO: Claude Code Generation Engine
## FROM: Project Lead  
## SUBJECT: Mandate for "KnowledgeCore Engine (K-Engine)" Development

---

## 1. 核心使命与设计哲学

您被委托协助开发 **KnowledgeCore Engine (K-Engine)**。这不是一个一次性脚本，而是我们团队AI能力的长期基础资产。所有生成的代码必须遵循以下核心理念：

### 1.1 设计原则

- **引擎，而非框架**：我们构建的是一个独立的高性能知识引擎。它必须与任何特定的前端应用（如Dify）解耦，并通过清晰、定义良好的API暴露其功能。

- **各取所长，输出统一**：针对每项任务，我们选择最佳可用工具。处理不同数据源的复杂性被隔离在初始解析阶段。该阶段的输出必须是统一的、高质量的Markdown格式，确保所有下游流程的一致性。

- **结构化，而非原始数据**：知识被视为结构化资产。原始文本块是不可接受的。每条知识都必须通过元数据进行增强，使其可发现、可理解且可信。

- **成本优先，性能为王**：优先使用国产大模型（DeepSeek/Qwen）和开源方案，在保证性能的前提下最大化降低成本。

- **测试驱动开发（TDD）**：严格遵循TDD方法论。每个功能模块必须先编写测试，定义预期行为，然后再实现功能代码。测试不仅是质量保证，更是设计工具。

### 1.2 开发方法论 - 测试驱动开发（TDD）

我们采用严格的TDD流程来确保代码质量和设计的合理性：

#### TDD核心流程（Red-Green-Refactor）

1. **Red（红灯）**：先编写一个失败的测试
   - 定义功能的预期行为
   - 测试必须清晰表达业务需求
   - 运行测试，确保它失败（因为功能还未实现）

2. **Green（绿灯）**：编写最少的代码使测试通过
   - 只实现让测试通过的最小功能
   - 不要过度设计或添加未测试的功能
   - 运行测试，确保通过

3. **Refactor（重构）**：优化代码结构
   - 在测试保护下重构代码
   - 消除重复，提高可读性
   - 确保测试仍然通过

#### TDD实践准则

- **测试优先**：任何新功能必须先有对应的测试用例
- **小步快跑**：每次只添加一个小功能，保持测试-实现循环简短
- **100%测试覆盖**：所有公共API必须有对应的测试
- **测试即文档**：测试用例应清晰展示如何使用功能
- **持续集成**：每次提交都必须通过所有测试

#### 测试层级

1. **单元测试**（Unit Tests）
   - 测试单个函数或类的行为
   - 使用mock隔离外部依赖
   - 执行速度快，反馈及时

2. **集成测试**（Integration Tests）
   - 测试模块间的交互
   - 使用真实的依赖（如数据库）
   - 确保组件协同工作

3. **端到端测试**（E2E Tests）
   - 测试完整的用户场景
   - 验证系统整体功能
   - 作为最终的质量保证

#### 开发确认流程

**重要**：在实现每个功能模块前，必须遵循以下流程：

1. **设计测试用例**：先展示测试代码设计，包括：
   - 测试文件结构
   - 主要测试场景
   - 预期的输入输出
   - Mock策略（如果需要）

2. **获得确认**：等待项目负责人确认测试设计合理

3. **实现测试**：编写完整的测试代码

4. **运行失败测试**：确保测试按预期失败

5. **获得实现许可**：再次获得确认后，才能开始实现功能

6. **实现功能**：编写使测试通过的最小代码

7. **重构优化**：在测试保护下改进代码质量

---

## 2. 完整的RAG生命周期：K-Engine的七大支柱

### 支柱1：摄入与解析（"前门"）

**目标**：接受任何文档格式并将其转换为单一、标准化、高质量的Markdown表示。

**关键工具**：
- **主要**：`LlamaParse` (API) 用于所有文档类型的统一解析

**核心逻辑**：
```python
async def master_parser(file_path: Path) -> Tuple[str, Dict[str, Any]]:
    """主解析器：根据文件类型路由到适当的专门解析器"""
    # 输出始终是 (markdown_string, initial_metadata)
    pass
```

### 支柱2：逻辑分块（"蓝图"）

**目标**：将统一的Markdown分解为语义有意义、上下文感知的块。

**关键工具**：
- `llama_index.core.node_parser.MarkdownNodeParser`
- `llama_index.core.node_parser.HierarchicalNodeParser`

**核心逻辑**：
```python
def intelligent_chunking(markdown: str, metadata: Dict) -> List[TextNode]:
    """基于文档逻辑结构的智能分块"""
    # 保留层级上下文在元数据中
    pass
```

### 支柱3：元数据增强（"智能层"）

**目标**：用机器生成的理解层丰富每个块，将其从原始文本转换为智能"知识资产"。

**关键工具**：
- LLM API (DeepSeek/Qwen)
- `Pydantic` 用于结构化输出

**核心逻辑**：
```python
class ChunkMetadata(BaseModel):
    summary: str  # 一句话摘要
    questions: List[str]  # 3-5个潜在问题
    chunk_type: str  # 分类标签
    keywords: List[str]  # 关键词提取
```

### 支柱4：嵌入与索引（"图书馆目录"）

**目标**：将增强的知识资产转换为可搜索的向量并高效存储。

**关键工具**：
- 嵌入模型：`qwen3-text-embedding-v3`（通过DashScope）
- 向量数据库：`ChromaDB`

**核心逻辑**：
```python
def multi_vector_indexing(chunk: TextNode) -> None:
    """多向量索引策略：content + summary + questions"""
    combined_text = f"{chunk.text}\n{chunk.metadata['summary']}\n{' '.join(chunk.metadata['questions'])}"
    # 嵌入并存储
    pass
```

### 支柱5：高级检索（"专家图书管理员"）

**目标**：为任何给定查询检索最相关、最高保真度的上下文。

**关键工具**：
- ChromaDB客户端
- 重排序模型：`bge-reranker-v2-m3-qwen`

**核心逻辑**：
```python
async def hybrid_retrieval(query: str, top_k: int = 5) -> List[TextNode]:
    """多阶段检索流程"""
    # 1. 查询转换（可选）
    # 2. 混合召回（向量 + BM25）
    # 3. 重排序
    pass
```

### 支柱6：生成（"发言人"）

**目标**：将检索到的上下文综合成清晰、准确、有帮助的答案，并附带引用。

**关键工具**：
- 主LLM：DeepSeek-V3 / Qwen2.5-72B

**核心逻辑**：
```python
def generate_with_citations(query: str, contexts: List[TextNode]) -> str:
    """基于检索内容生成答案，必须包含引用"""
    pass
```

### 支柱7：评估（"期末考试"）

**目标**：持续客观地衡量整个RAG系统的性能。

**关键工具**：
- `Ragas` 框架（配置使用DeepSeek作为评估LLM）
- 内部"黄金测试集"

**核心逻辑**：
```python
def evaluate_system(test_cases: List[Dict]) -> Dict[str, float]:
    """自动化评估流水线"""
    # 测量：Context Precision/Recall, Faithfulness, Answer Relevancy
    pass
```

---

## 3. 项目结构与标准

### 3.1 目录结构

```
knowledge-core-engine/
├── pyproject.toml              # 现代Python项目配置
├── .env.example               # 环境变量示例
├── src/
│   └── knowledge_core_engine/
│       ├── __init__.py
│       ├── api/               # FastAPI应用
│       │   ├── __init__.py
│       │   ├── app.py        # 主应用入口
│       │   ├── endpoints/    # API端点
│       │   └── models/       # Pydantic模型
│       ├── core/             # 核心业务逻辑
│       │   ├── __init__.py
│       │   ├── parsing/      # 解析模块
│       │   ├── chunking/     # 分块模块
│       │   ├── enhancement/  # 元数据增强
│       │   ├── embedding/    # 嵌入模块
│       │   ├── retrieval/    # 检索模块
│       │   └── generation/   # 生成模块
│       ├── pipelines/        # 高级流水线
│       │   ├── __init__.py
│       │   ├── ingestion.py  # 摄入流水线
│       │   └── evaluation.py # 评估流水线
│       ├── storage/          # 存储抽象层
│       │   ├── __init__.py
│       │   ├── vector_store.py
│       │   └── document_store.py
│       ├── evaluation/       # 评估框架
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── golden_set.py
│       └── utils/           # 工具函数
│           ├── __init__.py
│           ├── config.py    # 配置管理
│           └── logger.py    # 日志配置
├── tests/                   # 测试套件
│   ├── unit/               # 单元测试
│   ├── integration/        # 集成测试
│   └── e2e/               # 端到端测试
├── data/                   # 数据目录
│   ├── source_docs/       # 源文档
│   ├── golden_set/        # 评估数据集
│   └── cache/            # 缓存目录
├── docs/                  # 项目文档
├── examples/              # 示例代码
├── scripts/              # 实用脚本
├── .gitignore
├── README.md
└── CLAUDE.md            # 本文档
```

### 3.2 编码标准

- **必须**遵循PEP8风格指南
- 使用`ruff`进行代码检查和格式化（替代black+flake8）
- 所有函数和类必须有类型注解
- 所有公共API必须有docstring
- 测试覆盖率必须保持在80%以上

### 3.3 技术栈

```yaml
# 核心框架
framework: llama_index
api_framework: fastapi

# 文档处理
parsing:
  primary: llama_parse
  fallback: [pymupdf, python-docx, beautifulsoup4]

# 向量化
embedding:
  provider: dashscope
  model: text-embedding-v3
  dimensions: 1536

# 存储
vector_database: chromadb
document_store: sqlite  # 用于元数据

# 检索增强
reranker:
  model: bge-reranker-v2-m3-qwen
  provider: huggingface

# 生成
llm:
  primary: deepseek-v3
  fallback: qwen2.5-72b-instruct
  temperature: 0.1  # 知识库场景需要准确性

# 评估
evaluation:
  framework: ragas
  llm: deepseek-chat  # 评估用LLM
  
# 开发工具
linting: ruff
type_checking: mypy
testing: pytest
ci_cd: github_actions
```

---

## 4. 实施路线图

### 阶段1：基础设施（第1-2周）
- [ ] 项目初始化和环境配置
- [ ] 实现核心抽象和接口定义
- [ ] 设置日志、配置和错误处理

### 阶段2：核心功能（第3-4周）
- [ ] 实现解析和分块模块
- [ ] 完成元数据增强流水线
- [ ] 构建嵌入和索引系统

### 阶段3：检索与生成（第5-6周）
- [ ] 实现混合检索策略
- [ ] 集成重排序模型
- [ ] 完成生成模块和引用系统

### 阶段4：评估与优化（第7-8周）
- [ ] 建立评估框架
- [ ] 创建黄金测试集
- [ ] 性能优化和错误处理

### 阶段5：生产化（第9-10周）
- [ ] API开发和文档
- [ ] 部署脚本和Docker化
- [ ] 监控和日志系统

---

## 5. 关键性能指标（KPI）

- **延迟**：P95 < 3秒（端到端）
- **准确性**：Faithfulness > 0.9
- **相关性**：Answer Relevancy > 0.85
- **成本**：< ¥0.1/查询
- **吞吐量**：> 100 QPS

---

## 6. 特别注意事项

1. **LlamaParse配额**：每天1000次免费调用，需要实现缓存机制
2. **成本控制**：优先使用本地模型和缓存，避免重复调用API
3. **数据安全**：敏感文档必须本地处理，不上传到第三方API
4. **可扩展性**：所有模块必须支持异步操作和批处理
5. **监控告警**：实现完整的错误追踪和性能监控

---

## 7. 开发者指南

### 7.1 快速开始

```bash
# 克隆仓库
git clone <repo-url>
cd knowledge-core-engine

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入API密钥

# 运行测试
pytest

# 启动开发服务器
uvicorn knowledge_core_engine.api.app:app --reload
```

### 7.2 提交规范

使用约定式提交（Conventional Commits）：
- `feat:` 新功能
- `fix:` 错误修复
- `docs:` 文档更新
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建过程或辅助工具的变动

---

## 8. 开发者注意事项

### 8.1 测试与安全性

- **严禁任何测试脚本**：必要时走测试用例
- 所有测试必须遵循项目的测试流程
- 测试用例必须经过严格审查和批准

---

此文档是项目的根本指导原则。遵守这些原则不是可选的，而是我们成功的必要条件。

让我们开始构建下一代知识引擎！