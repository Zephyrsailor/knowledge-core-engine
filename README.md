# KnowledgeCore Engine (K-Engine)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-305%20passing-brightgreen)](tests/)
[![Code Style](https://img.shields.io/badge/Code%20Style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

🚀 **企业级RAG知识引擎** - 构建准确、可追溯、高性能的知识问答系统

[快速开始](#快速开始) | [核心特性](#核心特性) | [安装指南](#安装指南) | [评测优化](#评测与优化) | [API文档](#rest-api-服务)

</div>

---

## 🌟 项目简介

KnowledgeCore Engine（简称K-Engine）是一个专为企业设计的**高性能RAG（检索增强生成）知识引擎**。它通过结合先进的文档处理、智能检索和精准生成技术，帮助企业构建可靠、可追溯的知识问答系统。

### 🎯 核心价值

- **准确性高**：基于真实文档生成答案，大幅减少AI幻觉
- **可追溯性**：每个答案都提供明确的引用来源
- **易于集成**：提供简洁的Python API和REST API
- **成本优化**：优先使用国产模型，显著降低使用成本
- **高度可扩展**：模块化设计，支持自定义各个组件

## 快速开始

```python
from knowledge_core_engine import KnowledgeEngine
import asyncio

async def main():
    # 创建引擎
    engine = KnowledgeEngine()
    
    # 添加文档
    await engine.add("data/source_docs/")
    
    # 提问
    answer = await engine.ask("什么是RAG？")
    print(answer)

asyncio.run(main())
```

就是这么简单！🎉

> **注意**：确保您已经在 `.env` 文件中配置了API密钥，或通过环境变量设置。详见[环境变量配置](#环境变量配置)。

## 核心特性

### 📄 智能文档处理
- 支持多种格式：PDF、Word、Markdown、TXT等
- 使用LlamaParse进行高质量文档解析
- 智能分块策略，保持语义完整性
- 自动元数据增强，提升检索效果

### 🔍 高效检索系统
- **混合检索**：结合语义搜索和关键词匹配
- **灵活的BM25支持**：BM25S（轻量级）、Elasticsearch（企业级）
- **多种重排序选择**：本地模型（BGE、Qwen）、API服务（DashScope、Cohere）
- **支持多种向量数据库**：ChromaDB、Pinecone、Weaviate
- **智能查询扩展**：提升检索召回率

### 💡 精准答案生成
- 集成多种LLM：DeepSeek、通义千问、OpenAI
- 自动引用标注，支持多种引用格式
- 流式生成支持，提升用户体验
- 链式思考（CoT）和自我批判机制

### 📊 评估与监控
- 内置评估框架，支持多维度指标
- 性能监控和使用统计
- A/B测试支持
- 完整的日志和追踪

## 安装指南

### 环境要求

- Python 3.11+
- 2GB+ RAM
- 10GB+ 磁盘空间（用于向量存储）

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/Zephyrsailor/knowledge-core-engine.git
cd knowledge-core-engine

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装核心依赖
pip install -e .

# 可选：安装额外功能
pip install -e ".[reranker-hf]"    # 安装HuggingFace重排序模型支持
pip install -e ".[elasticsearch]"   # 安装Elasticsearch支持
pip install -e ".[dev]"            # 安装开发依赖

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 环境变量配置

在`.env`文件中配置以下变量：

```bash
# LLM配置（选择其一）
DEEPSEEK_API_KEY=your_deepseek_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key  # 用于通义千问
OPENAI_API_KEY=your_openai_api_key  # 可选

# 文档解析
LLAMA_CLOUD_API_KEY=your_llama_parse_key  # 可选，提供1000次/天免费额度

# 向量数据库（ChromaDB默认无需配置）
# PINECONE_API_KEY=your_pinecone_key  # 如使用Pinecone
# WEAVIATE_URL=http://localhost:8080  # 如使用Weaviate

# 日志配置（可选）
# LOG_LEVEL=DEBUG  # 可选值: DEBUG, INFO, WARNING, ERROR
```

## 使用示例

```python
import asyncio
from knowledge_core_engine import KnowledgeEngine

async def main():
    # 创建知识引擎
    engine = KnowledgeEngine()
    
    # 添加文档
    await engine.add("data/source_docs/")
    
    # 提问
    answer = await engine.ask("什么是RAG技术？")
    print(answer)

# 运行
asyncio.run(main())
```

### 更多使用示例

#### 1. 获取详细信息

```python
# 获取详细的答案信息
result = await engine.ask("什么是RAG？", return_details=True)

print(f"答案: {result['answer']}")
print(f"引用: {result['citations']}")
print(f"上下文: {result['contexts']}")
```

#### 2. 搜索功能

```python
# 搜索相关文档片段
results = await engine.search("检索增强", top_k=10)

for result in results:
    print(f"相关度: {result['score']:.3f}")
    print(f"内容: {result['content'][:100]}...")
```


## 高级功能

### 高级检索功能

K-Engine 提供了一系列高级检索功能，让您可以根据需求进行精细化配置：

#### 检索策略

```python
# 1. 纯向量检索（适合语义相似度匹配）
engine = KnowledgeEngine(
    retrieval_strategy="vector"
)

# 2. 纯关键词检索（适合精确匹配）
engine = KnowledgeEngine(
    retrieval_strategy="bm25"
)

# 3. 混合检索（默认，结合两者优势）
engine = KnowledgeEngine(
    retrieval_strategy="hybrid",
    vector_weight=0.7,  # 向量检索权重
    bm25_weight=0.3,    # BM25检索权重
    fusion_method="weighted"  # 可选: weighted, rrf
)
```


#### 重排序

K-Engine 支持多种重排序方式，满足不同场景需求：

```python
# 1. 使用本地 BGE 模型（推荐）
engine = KnowledgeEngine(
    enable_reranking=True,
    reranker_provider="huggingface",
    reranker_model="bge-reranker-v2-m3",
    use_fp16=True,  # 节省内存
    rerank_top_k=5
)

# 2. 使用本地 Qwen 模型（精度更高，适合36GB内存）
engine = KnowledgeEngine(
    enable_reranking=True,
    reranker_provider="huggingface",
    reranker_model="qwen3-reranker-8b",
    use_fp16=True,
    rerank_top_k=5
)

# 3. 使用 API 服务（无需本地资源）
engine = KnowledgeEngine(
    enable_reranking=True,
    reranker_provider="api",
    reranker_api_provider="dashscope",
    reranker_model="gte-rerank-v2",
    rerank_top_k=5
)
```


### 分块策略配置

```python
# 层级分块（保留文档结构）
engine = KnowledgeEngine(
    enable_hierarchical_chunking=True,
    chunk_size=1024,
    chunk_overlap=128
)

# 语义分块（默认）
engine = KnowledgeEngine(
    enable_semantic_chunking=True,
    chunk_size=512,
    chunk_overlap=50
)
```

### 完整配置选项

```python
engine = KnowledgeEngine(
    # 基础配置
    llm_provider="deepseek",             # 可选: deepseek, qwen, openai
    embedding_provider="dashscope",       # 可选: dashscope, openai
    persist_directory="./data/kb",        # 持久化目录
    log_level="INFO",                    # 日志级别: DEBUG, INFO, WARNING, ERROR
    
    # 分块配置
    enable_hierarchical_chunking=False,   # 层级分块
    enable_semantic_chunking=True,        # 语义分块
    chunk_size=512,                       # 分块大小
    chunk_overlap=50,                     # 分块重叠
    enable_metadata_enhancement=False,    # 元数据增强
    
    # 检索配置
    retrieval_strategy="hybrid",          # 可选: vector, bm25, hybrid
    retrieval_top_k=10,                   # 检索文档数量
    vector_weight=0.7,                    # 向量检索权重
    bm25_weight=0.3,                      # BM25权重
    
    # 查询扩展
    enable_query_expansion=False,         # 启用查询扩展
    query_expansion_method="llm",         # 扩展方法: llm
    query_expansion_count=3,              # 扩展数量
    
    # 重排序
    enable_reranking=False,               # 启用重排序
    reranker_provider="huggingface",      # 可选: huggingface, api
    reranker_model="bge-reranker-v2-m3",  # 重排序模型
    reranker_api_provider=None,           # API提供商: dashscope
    use_fp16=True,                        # 使用半精度（节省内存）
    rerank_top_k=5,                       # 重排后文档数
    
    # 其他配置
    include_citations=True,               # 是否包含引用
)
```

### 高级使用示例

```python
# 高性能配置示例
engine = KnowledgeEngine(
    # 使用层级分块保留文档结构
    enable_hierarchical_chunking=True,
    enable_metadata_enhancement=True,
    
    # 混合检索
    retrieval_strategy="hybrid",
    
    # 启用查询扩展
    enable_query_expansion=True,
    
    # 使用重排序
    enable_reranking=True,
    reranker_model="bge-reranker-v2-m3",
    
    # 优化参数
    chunk_size=1024,
    retrieval_top_k=20,  # 初始检索更多文档
    rerank_top_k=5       # 重排后保留最相关的5个
)

# 添加文档
result = await engine.add("docs/")
print(f"处理了 {result['total_chunks']} 个文档块")

# 智能问答
answer = await engine.ask(
    "RAG技术的主要优势是什么？",
    return_details=True
)

print(f"答案: {answer['answer']}")
print(f"使用了 {len(answer['contexts'])} 个相关文档")
print(f"引用: {answer['citations']}")
```

## REST API 服务

K-Engine提供了完整的REST API，方便集成到各种应用中。

### 启动API服务器

```bash
# 完整功能的API服务器（推荐）
python examples/api_server_simple.py

# 最小化API（仅健康检查）
uvicorn knowledge_core_engine.api.app:app --host 0.0.0.0 --port 8000
```

> 注意：`examples/api_server.py` 提供了完整的RAG功能API，包括文档上传、查询、流式响应等。
> 而 `knowledge_core_engine.api.app` 只是一个最小化的入口点。

### API端点示例

#### 上传文档
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "accept: application/json" \
  -F "file=@/path/to/document.pdf"
```

#### 查询知识库
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是RAG技术？",
    "top_k": 5,
    "include_citations": true
  }'
```

#### 流式查询
```javascript
const eventSource = new EventSource('http://localhost:8000/query/stream');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.content) {
        console.log(data.content);
    }
};
```

## 评测与优化

K-Engine 提供了完善的评测框架，帮助您评估和优化RAG系统的性能。

### 快速开始评测

```bash
# 运行评测（默认测试10个样本）
python scripts/run_evaluation.py

# 指定测试样本数
python scripts/run_evaluation.py --samples 5

# 查看帮助
python scripts/run_evaluation.py --help
```

### 评测流程说明

1. **准备知识库**：从测试数据集中提取内容，创建评测用知识库
2. **生成答案**：对每个测试问题运行完整的RAG流程
3. **计算指标**：评估答案质量和系统性能
4. **生成报告**：输出详细的评测结果

### 评测指标

- **成功率**：成功生成答案的比例
- **关键词覆盖率**：答案中包含标准答案关键词的比例（目标 > 60%）
- **平均答案长度**：生成答案的平均字符数
- **评测耗时**：完成评测所需时间

### 黄金测试集

项目包含两个精心设计的测试集：
- `data/golden_set/rag_qa_dataset.json` - RAG技术相关的10个测试用例
- `data/golden_set/rag_test_set.json` - 更多样化的测试场景
- 每个用例包含：问题、标准答案、理想检索内容、元数据

### 评测结果示例

```json
{
  "summary": {
    "total_cases": 2,
    "successful_cases": 2,
    "success_rate": 1.0,
    "avg_keyword_coverage": 0.82,  // 82%的关键词覆盖率
    "avg_answer_length": 646.0
  },
  "metadata": {
    "config_profile": "default",
    "duration": 37.5  // 总耗时（秒）
  }
}
```

### 优化建议

1. **如果关键词覆盖率低**（<60%）
   - 增加 `retrieval_top_k` 获取更多上下文
   - 启用混合检索策略 `retrieval_strategy="hybrid"`
   - 调整 BM25 权重以提高关键词匹配

2. **如果生成速度慢**
   - 使用更快的模型（如 qwen-turbo）
   - 减少 `max_tokens` 限制
   - 关闭重排序或查询扩展功能

3. **如果答案质量不佳**
   - 启用元数据增强 `enable_metadata_enhancement=True`
   - 使用更大的分块大小 `chunk_size=1024`
   - 启用层级分块保留文档结构

### 评测结果存储

- **位置**：`evaluation_results/` 目录
- **格式**：`evaluation_default_YYYYMMDD_HHMMSS.json`
- **内容**：包含完整的测试结果和性能指标

### 持续改进建议

1. **定期评测**：每次重大更新后运行评测
2. **基准对比**：保存基准结果用于对比改进
3. **逐步优化**：先用少量样本测试，确认后再全量评测
4. **关注趋势**：跟踪关键指标的变化趋势

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   文档输入   │ ──▶ │  解析模块   │ ──▶ │  分块模块   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  生成模块   │ ◀── │  检索模块   │ ◀── │  向量存储   │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│ 答案 + 引用 │
└─────────────┘
```

### 核心模块说明

1. **解析模块**：使用LlamaParse将各种格式文档转换为统一的Markdown格式
2. **分块模块**：智能分割文档，保持语义完整性
3. **向量存储**：将文本转换为向量并高效存储
4. **检索模块**：混合检索策略，快速找到相关内容
5. **生成模块**：基于检索结果生成准确答案


## 性能优化建议

### 1. 文档处理优化
- 批量处理文档以提高效率
- 使用异步处理充分利用IO
- 合理设置分块大小（建议256-1024）

### 2. 检索优化
- **混合检索策略**：结合向量检索和BM25关键词检索
- **重排序优化**：BGE-reranker-v2-m3（平衡）、DashScope API（低成本）
- **合理设置top_k**：初始检索15-20个，重排后保留3-5个

### 3. 生成优化
- 使用流式生成改善响应时间
- 选择合适的温度参数
- 启用缓存减少重复计算

### 4. 成本优化
- 优先使用国产模型（DeepSeek/Qwen）
- 合理设置max_tokens
- 使用本地向量数据库（ChromaDB）

## 测试

运行测试套件：

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/unit/core/generation/

# 查看测试覆盖率
pytest --cov=knowledge_core_engine --cov-report=html
```


## 贡献指南

我们欢迎所有形式的贡献！请查看[CONTRIBUTING.md](CONTRIBUTING.md)了解详情。

### 开发流程

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

### 代码规范

- 使用 `ruff` 进行代码格式化和检查
- 遵循 TDD（测试驱动开发）原则
- 所有公共API必须有文档字符串
- 提交信息遵循[约定式提交](https://www.conventionalcommits.org/)

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [LlamaIndex](https://github.com/jerryjliu/llama_index) - 核心框架
- [LlamaParse](https://github.com/run-llama/llama_parse) - 文档解析
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量存储
- [DeepSeek](https://www.deepseek.com/) - LLM提供商
- [DashScope](https://dashscope.aliyun.com/) - 嵌入和LLM服务

## 更多文档

- [配置指南](docs/CONFIGURATION_GUIDE.md) - 详细的配置系统说明
- [高级功能](docs/ADVANCED_FEATURES.md) - 深入了解高级特性
- [专业评测指南](docs/EVALUATION_GUIDE.md) - 完整的RAG系统评测方案
- [检索架构](docs/RETRIEVAL_ARCHITECTURE.md) - 深入理解检索系统设计


---

