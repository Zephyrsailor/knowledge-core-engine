# KnowledgeCore Engine (K-Engine)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-264%20passing-brightgreen)](tests/)
[![Code Style](https://img.shields.io/badge/Code%20Style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

🚀 **企业级RAG知识引擎** - 构建准确、可追溯、高性能的知识问答系统

[快速开始](#快速开始) | [核心特性](#核心特性) | [安装指南](#安装指南) | [使用示例](#使用示例) | [API文档](#api文档)

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

## 🚀 核心特性

### 📄 智能文档处理
- 支持多种格式：PDF、Word、Markdown、TXT等
- 使用LlamaParse进行高质量文档解析
- 智能分块策略，保持语义完整性
- 自动元数据增强，提升检索效果

### 🔍 高效检索系统
- 混合检索：结合语义搜索和关键词匹配
- 支持多种向量数据库：ChromaDB、Pinecone、Weaviate
- 智能重排序，提升结果相关性
- 支持元数据过滤和高级查询

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

## 📦 安装指南

### 环境要求

- Python 3.8+
- 2GB+ RAM
- 10GB+ 磁盘空间（用于向量存储）

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/your-org/knowledge-core-engine.git
cd knowledge-core-engine

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

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
```

## 🎯 快速开始

### 1. 基础使用示例

```python
import asyncio
from pathlib import Path
from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.pipelines.ingestion import IngestionPipeline
from knowledge_core_engine.pipelines.retrieval import RetrievalPipeline
from knowledge_core_engine.pipelines.generation import GenerationPipeline

async def main():
    # 1. 配置系统
    config = RAGConfig(
        llm_provider="deepseek",
        llm_api_key="your_api_key",
        embedding_provider="dashscope",
        embedding_api_key="your_api_key",
        vector_store_provider="chromadb",
        include_citations=True
    )
    
    # 2. 加载文档
    ingestion = IngestionPipeline(config)
    await ingestion.initialize()
    
    # 处理PDF文档
    result = await ingestion.process_document(Path("./docs/技术文档.pdf"))
    print(f"✅ 文档处理完成：{result['chunks_created']} 个知识片段")
    
    # 3. 提问并获取答案
    retrieval = RetrievalPipeline(config)
    generation = GenerationPipeline(config)
    await retrieval.initialize()
    await generation.initialize()
    
    query = "RAG技术的主要优势是什么？"
    
    # 检索相关内容
    contexts = await retrieval.retrieve(query, top_k=5)
    
    # 生成答案
    result = await generation.generate(query, contexts)
    
    print(f"\n💡 问题：{query}")
    print(f"📝 答案：{result.answer}")
    print(f"📚 引用：{len(result.citations)} 个来源")

# 运行示例
asyncio.run(main())
```

### 2. 批量文档处理

```python
async def batch_process_documents(directory: Path, config: RAGConfig):
    """批量处理文档目录"""
    ingestion = IngestionPipeline(config)
    await ingestion.initialize()
    
    # 支持的文档格式
    supported_formats = ['.pdf', '.docx', '.md', '.txt']
    documents = []
    
    for format in supported_formats:
        documents.extend(directory.glob(f"*{format}"))
    
    print(f"📁 找到 {len(documents)} 个文档")
    
    # 批量处理
    for doc in documents:
        try:
            result = await ingestion.process_document(doc)
            print(f"✅ {doc.name}: {result['chunks_created']} 个片段")
        except Exception as e:
            print(f"❌ {doc.name}: 处理失败 - {e}")
```

### 3. 流式生成答案

```python
async def stream_answer(query: str, config: RAGConfig):
    """流式生成答案，提升用户体验"""
    # 初始化管道
    retrieval = RetrievalPipeline(config)
    generation = GenerationPipeline(config)
    await retrieval.initialize()
    await generation.initialize()
    
    # 检索上下文
    contexts = await retrieval.retrieve(query, top_k=5)
    
    # 流式生成
    print(f"💡 答案生成中：", end="", flush=True)
    
    async for chunk in generation.stream_generate(query, contexts):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        
        if chunk.is_final and chunk.citations:
            print(f"\n\n📚 引用来源：")
            for citation in chunk.citations:
                print(f"   [{citation.index}] {citation.document_title}")
```

## 📖 高级功能

### 1. 自定义提示模板

```python
from knowledge_core_engine.core.generation.prompt_builder import PromptTemplate

# 创建专业分析模板
technical_template = PromptTemplate(
    name="technical_analysis",
    template="""作为技术专家，请基于以下文档进行深入分析。

问题：{query}

参考文档：
{contexts}

请提供：
1. 技术原理解释
2. 实施步骤
3. 注意事项
4. 最佳实践

分析："""
)

# 使用自定义模板
result = await generation.generate(
    query="如何优化RAG系统性能？",
    contexts=contexts,
    template=technical_template.template
)
```

### 2. 高级检索配置

```python
# 混合检索配置
contexts = await retrieval.retrieve(
    query="机器学习算法",
    top_k=10,
    search_type="hybrid",
    hybrid_alpha=0.7,  # 70%语义搜索，30%关键词匹配
    filters={
        "document_type": {"$eq": "技术文档"},
        "year": {"$gte": 2023}
    }
)

# 使用重排序
contexts = await retrieval.retrieve_with_rerank(
    query="深度学习应用",
    top_k=20,
    rerank_top_k=5  # 从20个结果中重排序选出最相关的5个
)
```

### 3. 多语言支持

```python
# 中文配置
zh_config = RAGConfig(
    llm_provider="qwen",
    extra_params={"language": "zh"}
)

# 英文配置
en_config = RAGConfig(
    llm_provider="openai",
    extra_params={"language": "en"}
)

# 根据查询语言自动选择
async def multilingual_query(query: str):
    # 简单的语言检测
    is_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
    config = zh_config if is_chinese else en_config
    
    # 处理查询...
```

## 🌐 REST API 服务

K-Engine提供了完整的REST API，方便集成到各种应用中。

### 启动API服务器

```bash
# 开发模式
python examples/api_server.py

# 生产模式
uvicorn knowledge_core_engine.api.app:app --host 0.0.0.0 --port 8000
```

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

## 🏗️ 系统架构

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

## 🔧 配置选项

### RAGConfig 完整参数

```python
config = RAGConfig(
    # LLM配置
    llm_provider="deepseek",  # 可选: deepseek, qwen, openai
    llm_model="deepseek-chat",
    llm_api_key="your_key",
    temperature=0.1,  # 0-2，越低越保守
    max_tokens=2048,
    
    # 嵌入配置
    embedding_provider="dashscope",  # 可选: dashscope, openai
    embedding_model="text-embedding-v3",
    embedding_api_key="your_key",
    embedding_dimension=1536,
    
    # 向量存储配置
    vector_store_provider="chromadb",  # 可选: chromadb, pinecone, weaviate
    vector_store_path="./data/chroma_db",
    
    # 生成配置
    include_citations=True,
    
    # 高级参数
    extra_params={
        "language": "zh",  # zh或en
        "chunk_size": 512,  # 分块大小
        "chunk_overlap": 50,  # 分块重叠
        "enable_metadata_enhancement": True,  # 元数据增强
        "enable_reranking": True,  # 重排序
        "reranker_model": "bge-reranker-v2-m3",
        "citation_style": "inline",  # inline, footnote, endnote
        "enable_cot": False,  # 链式思考
        "max_retries": 3,  # 重试次数
        "temperature_decay": 0.1  # 重试时温度衰减
    }
)
```

## 📊 性能优化建议

### 1. 文档处理优化
- 批量处理文档以提高效率
- 使用异步处理充分利用IO
- 合理设置分块大小（建议256-1024）

### 2. 检索优化
- 使用混合检索提高召回率
- 启用重排序提升精度
- 合理设置top_k（建议3-10）

### 3. 生成优化
- 使用流式生成改善响应时间
- 选择合适的温度参数
- 启用缓存减少重复计算

### 4. 成本优化
- 优先使用国产模型（DeepSeek/Qwen）
- 合理设置max_tokens
- 使用本地向量数据库（ChromaDB）

## 🧪 测试

运行测试套件：

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/unit/core/generation/

# 查看测试覆盖率
pytest --cov=knowledge_core_engine --cov-report=html
```

当前测试状态：
- ✅ 264个测试全部通过
- 📊 测试覆盖率 > 80%

## 🤝 贡献指南

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

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [LlamaIndex](https://github.com/jerryjliu/llama_index) - 核心框架
- [LlamaParse](https://github.com/run-llama/llama_parse) - 文档解析
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量存储
- [DeepSeek](https://www.deepseek.com/) - LLM提供商
- [DashScope](https://dashscope.aliyun.com/) - 嵌入和LLM服务

## 📞 联系我们

- 项目主页：[GitHub](https://github.com/your-org/knowledge-core-engine)
- 问题反馈：[Issues](https://github.com/your-org/knowledge-core-engine/issues)
- 邮箱：contact@knowledge-core.ai

---

<div align="center">
Made with ❤️ by KnowledgeCore Team
</div>