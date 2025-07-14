# KnowledgeCore Engine 配置指南

本文档说明了 KnowledgeCore Engine 的配置系统架构。

## 配置系统概览

KnowledgeCore Engine 采用分层配置架构，不同层次的配置服务于不同的目的：

```
┌─────────────────────────────────────────┐
│          用户接口层                      │
│     KnowledgeEngine(配置参数)           │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│           核心配置层                     │
│    core.config.RAGConfig                │
│  (RAG相关的所有运行时配置)              │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────┴──────────────────────┐
│          环境配置层                      │
│    utils.config.Settings                │
│  (环境变量、API密钥、文件路径)          │
└─────────────────────────────────────────┘
```

## 1. 核心配置 (core.config.RAGConfig)

**文件位置**: `src/knowledge_core_engine/core/config.py`

**用途**: 这是系统的主要配置类，包含所有RAG相关的运行时配置。

### 主要配置项

```python
from knowledge_core_engine.core.config import RAGConfig

config = RAGConfig(
    # === 基础配置 ===
    llm_provider="deepseek",              # LLM提供商
    embedding_provider="dashscope",       # 嵌入模型提供商
    vectordb_provider="chromadb",         # 向量数据库
    
    # === 分块配置 ===
    enable_hierarchical_chunking=False,   # 层级分块
    enable_semantic_chunking=True,        # 语义分块
    chunk_size=512,                       # 分块大小
    chunk_overlap=50,                     # 分块重叠
    enable_metadata_enhancement=False,    # 元数据增强
    
    # === 检索配置 ===
    retrieval_strategy="hybrid",          # 检索策略
    retrieval_top_k=10,                   # 检索数量
    vector_weight=0.7,                    # 向量权重
    bm25_weight=0.3,                      # BM25权重
    
    # === 高级功能 ===
    enable_query_expansion=False,         # 查询扩展
    enable_reranking=False,               # 重排序
    include_citations=True                # 包含引用
)
```

### 使用场景

- 被 `KnowledgeEngine` 使用
- 被所有核心组件（Retriever、Generator、Embedder等）使用
- 支持运行时动态配置

## 2. 环境配置 (utils.config.Settings)

**文件位置**: `src/knowledge_core_engine/utils/config.py`

**用途**: 管理环境变量、API密钥和应用级设置。

### 主要配置项

```python
from knowledge_core_engine.utils.config import get_settings

settings = get_settings()

# 访问配置
api_key = settings.dashscope_api_key
log_level = settings.log_level
cache_dir = settings.cache_dir
```

### 配置来源

1. 环境变量
2. `.env` 文件
3. 默认值

### 示例 .env 文件

```bash
# API Keys
DASHSCOPE_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
LLAMA_CLOUD_API_KEY=your_key_here

# Paths
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CACHE_DIR=./data/cache

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

## 3. 废弃的配置 (config.py)

**文件位置**: `src/knowledge_core_engine/config.py`

**状态**: **已废弃** - 请勿使用

这个文件是早期版本的遗留，已被标记为废弃。请使用 `core.config.RAGConfig` 代替。

## 4. Pipeline配置 (PipelineConfig)

**文件位置**: `src/knowledge_core_engine/core/rag_pipeline.py`

**用途**: 专门用于基于提供商的 RAGPipeline 实现（实验性功能）。

**注意**: 对于大多数用例，推荐使用 `KnowledgeEngine` 而不是 `RAGPipeline`。

## 配置最佳实践

### 1. 简单使用场景

```python
# 最简单的方式 - 使用默认配置
engine = KnowledgeEngine()

# 或者只配置必要的参数
engine = KnowledgeEngine(
    llm_provider="qwen",
    persist_directory="./my_knowledge_base"
)
```

### 2. 高级使用场景

```python
# 创建自定义配置
config = RAGConfig(
    # 基础配置
    llm_provider="deepseek",
    temperature=0.1,
    
    # 启用高级功能
    enable_hierarchical_chunking=True,
    enable_query_expansion=True,
    enable_reranking=True,
    
    # 性能优化
    retrieval_top_k=20,
    rerank_top_k=5
)

# 使用配置创建引擎
engine = KnowledgeEngine(**config.__dict__)
```

### 3. 环境变量管理

```python
# 在代码中访问环境设置
from knowledge_core_engine.utils.config import get_settings

settings = get_settings()

# 使用设置
if settings.enable_cache:
    cache_path = settings.cache_dir / "my_cache"
```

## 配置优先级

1. **显式参数** > **环境变量** > **默认值**
2. `KnowledgeEngine` 构造函数参数会覆盖所有其他配置
3. API密钥优先从环境变量加载

## 常见问题

### Q: 应该使用哪个配置类？

A: 
- 使用 `RAGConfig` 进行RAG相关配置
- 使用 `get_settings()` 获取环境设置
- 不要使用废弃的 `config.py`

### Q: 如何添加新的配置项？

A: 
1. 如果是RAG相关的，添加到 `core.config.RAGConfig`
2. 如果是环境相关的，添加到 `utils.config.Settings`
3. 记得更新验证逻辑和默认值

### Q: RAGPipeline 和 KnowledgeEngine 的区别？

A: 
- `KnowledgeEngine`: 推荐使用，高级接口，功能丰富
- `RAGPipeline`: 实验性，提供更底层的控制，需要更多手动配置

## 未来计划

1. 移除废弃的 `config.py` 文件
2. 考虑合并 `RAGPipeline` 到 `KnowledgeEngine` 或明确分离使用场景
3. 添加配置验证工具
4. 支持配置热重载

---

更多信息请参考[主文档](../README.md)或查看[高级功能文档](ADVANCED_FEATURES.md)。