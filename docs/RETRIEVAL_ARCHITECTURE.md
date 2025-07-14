# KnowledgeCore Engine 检索系统架构

本文档介绍了 KnowledgeCore Engine 的灵活检索架构，支持多种 BM25 和 Reranker 实现。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    用户查询 (Query)                      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────┐
│                   检索策略 (Strategy)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Vector    │  │    BM25     │  │   Hybrid    │    │
│  │  (向量检索)  │  │ (关键词检索) │  │  (混合检索)  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────┐
│                  BM25 Providers                          │
│  ┌─────────────┐  ┌─────────────────┐                  │
│  │   BM25S     │  │  Elasticsearch  │                  │
│  │ (轻量级)    │  │    (企业级)     │                  │
│  └─────────────┘  └─────────────────┘                  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────┐
│                 Reranker Providers                       │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │   HuggingFace    │  │      API        │            │
│  │ (BGE, Qwen等)    │  │ (DashScope等)   │            │
│  └──────────────────┘  └──────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

## 1. BM25 Provider 系统

### 1.1 BM25S (默认推荐)

**特点**：
- 纯 Python 实现，无需外部服务
- 比传统 rank-bm25 快 500 倍
- 支持中文、英文等多语言
- 轻量级依赖（仅需 numpy, scipy）

**使用场景**：
- 中小规模数据集（< 100万文档）
- 快速原型开发
- 无需分布式检索的场景

**配置示例**：
```python
config = RAGConfig(
    retrieval_strategy="hybrid",
    bm25_provider="bm25s",
    language="zh",  # 支持 en, zh, multi
    bm25_k1=1.5,
    bm25_b=0.75
)
```

### 1.2 Elasticsearch

**特点**：
- 业界标准的搜索引擎
- 支持分布式、高并发
- 丰富的查询 DSL
- 实时索引更新

**使用场景**：
- 大规模生产环境
- 需要复杂查询功能
- 需要实时更新索引
- 已有 Elasticsearch 基础设施

**配置示例**：
```python
config = RAGConfig(
    retrieval_strategy="hybrid",
    bm25_provider="elasticsearch",
    elasticsearch_url="http://localhost:9200",
    elasticsearch_index="knowledge_core",
    elasticsearch_username="elastic",  # 可选
    elasticsearch_password="password"   # 可选
)
```

## 2. Reranker Provider 系统

### 2.1 HuggingFace Provider

支持多种本地模型，适合有 GPU 资源的场景。

#### BGE 系列模型

**模型选择**：
- `bge-reranker-v2-m3`：多语言支持，性能均衡（推荐）
- `bge-reranker-large`：英文效果最佳，资源占用较高
- `bge-reranker-base`：轻量级选择

**后端支持**：
- FlagEmbedding（推荐）：官方实现，支持 fp16
- sentence-transformers：备选方案

#### Qwen3 系列模型

**模型选择**：
- `qwen3-reranker-8b`：最高精度，需要 16GB+ 内存
- `qwen3-reranker-4b`：平衡选择
- `qwen3-reranker-0.6b`：轻量级

**特点**：
- 使用 yes/no token 概率计算相关性
- 支持超长文本（32k tokens）
- 支持自定义指令优化

**配置示例**：
```python
# BGE 模型
config = RAGConfig(
    enable_reranking=True,
    reranker_provider="huggingface",
    reranker_model="bge-reranker-v2-m3",
    use_fp16=True,  # 节省内存
    reranker_device="cuda"  # 或 "cpu"
)

# Qwen 模型（适合您的 36GB Mac）
config = RAGConfig(
    enable_reranking=True,
    reranker_provider="huggingface",
    reranker_model="qwen3-reranker-8b",
    use_fp16=True
)
```

### 2.2 API Provider

通过 API 调用重排序服务，无需本地资源。

#### 支持的 API 服务

1. **DashScope (推荐)**
   - 模型：`gte-rerank-v2`
   - 支持 50+ 语言
   - 单次最多 500 文档
   - 按需付费

2. **Cohere**
   - 模型：`rerank-english-v2.0`
   - 英文效果优秀
   - 单次最多 1000 文档

3. **Jina**
   - 模型：`jina-reranker-v1-base-en`
   - 轻量级选择

**配置示例**：
```python
# DashScope API
config = RAGConfig(
    enable_reranking=True,
    reranker_provider="api",
    reranker_api_provider="dashscope",
    reranker_model="gte-rerank-v2"
)

# Cohere API
config = RAGConfig(
    enable_reranking=True,
    reranker_provider="api",
    reranker_api_provider="cohere",
    reranker_model="rerank-english-v2.0"
)
```

## 3. 检索策略组合

### 3.1 高性能本地配置

适合有充足硬件资源的场景：

```python
config = RAGConfig(
    # 混合检索
    retrieval_strategy="hybrid",
    bm25_provider="bm25s",
    vector_weight=0.7,
    bm25_weight=0.3,
    
    # 本地重排序
    enable_reranking=True,
    reranker_provider="huggingface",
    reranker_model="qwen3-reranker-8b",
    use_fp16=True,
    
    # 查询扩展
    enable_query_expansion=True,
    query_expansion_method="llm"
)
```

### 3.2 云端 API 配置

适合资源受限或快速部署：

```python
config = RAGConfig(
    # 混合检索
    retrieval_strategy="hybrid",
    bm25_provider="bm25s",
    
    # API 重排序
    enable_reranking=True,
    reranker_provider="api",
    reranker_api_provider="dashscope",
    
    # 成本优化
    retrieval_top_k=20,
    rerank_top_k=5
)
```

### 3.3 企业级配置

适合生产环境：

```python
config = RAGConfig(
    # Elasticsearch 检索
    retrieval_strategy="hybrid",
    bm25_provider="elasticsearch",
    elasticsearch_url="http://es-cluster:9200",
    
    # 本地重排序
    enable_reranking=True,
    reranker_provider="huggingface",
    reranker_model="bge-reranker-v2-m3",
    
    # 性能优化
    retrieval_top_k=50,
    rerank_top_k=10
)
```

## 4. 性能对比

| 配置 | 延迟 | 准确性 | 资源需求 | 成本 |
|------|------|--------|----------|------|
| BM25S + BGE-M3 | 中 | 高 | 中 | 低 |
| BM25S + Qwen3-8B | 高 | 最高 | 高 | 低 |
| BM25S + DashScope API | 低 | 高 | 低 | 中 |
| Elasticsearch + BGE | 低 | 高 | 高 | 中 |
| 纯向量检索 | 最低 | 中 | 低 | 低 |

## 5. 最佳实践

### 5.1 选择 BM25 Provider

- **开发测试**：使用 BM25S
- **生产环境**：
  - 数据量 < 100万：BM25S
  - 数据量 > 100万或需要实时更新：Elasticsearch

### 5.2 选择 Reranker

- **精度优先**：Qwen3-8B（本地）或 DashScope API
- **速度优先**：BGE-reranker-base 或不使用重排序
- **平衡选择**：BGE-reranker-v2-m3

### 5.3 资源估算

- **BM25S**：< 1GB 内存
- **BGE-M3**：约 2-4GB 内存
- **Qwen3-8B**：约 16GB 内存（fp16）
- **API 服务**：无本地资源需求

## 6. 故障排除

### 常见问题

1. **Elasticsearch 连接失败**
   ```
   检查 Elasticsearch 是否运行：
   curl http://localhost:9200
   ```

2. **模型加载失败**
   ```
   安装必要依赖：
   pip install "knowledge-core-engine[reranker-hf]"
   ```

3. **API 调用失败**
   ```
   检查 API 密钥配置：
   export DASHSCOPE_API_KEY=your_key
   ```

## 7. 未来规划

- 支持更多 BM25 实现（如 Tantivy）
- 支持更多 Reranker 模型
- 自动选择最优配置
- 性能基准测试工具