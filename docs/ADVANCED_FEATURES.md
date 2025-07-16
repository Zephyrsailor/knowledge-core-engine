# 高级功能详解

本文档详细介绍了 KnowledgeCore Engine 的高级功能和配置选项。

## 目录

1. [智能分块策略](#智能分块策略)
2. [高级检索功能](#高级检索功能)
3. [查询优化技术](#查询优化技术)
4. [性能优化配置](#性能优化配置)
5. [最佳实践建议](#最佳实践建议)

## 智能分块策略

### 层级分块 (Hierarchical Chunking)

层级分块保留了文档的结构信息，特别适合处理有明确章节结构的文档。

```python
engine = KnowledgeEngine(
    enable_hierarchical_chunking=True,
    chunk_size=1024,
    chunk_overlap=128
)
```

**特点：**
- 保留父子关系和兄弟关系
- 维护文档的层级路径
- 支持基于结构的检索

**适用场景：**
- 技术文档
- 学术论文
- 法律文件
- 带有明确章节的书籍

### 语义分块 (Semantic Chunking)

语义分块根据内容的语义边界进行分割，确保每个块都包含完整的语义信息。

```python
engine = KnowledgeEngine(
    enable_semantic_chunking=True,
    chunk_size=512,
    chunk_overlap=50
)
```

**特点：**
- 基于句子和段落边界分割
- 避免在语义中间截断
- 保持上下文的完整性

**适用场景：**
- 新闻文章
- 博客文章
- 对话记录
- 一般性文本

### 元数据增强 (Metadata Enhancement)

使用 LLM 自动为每个文档块生成摘要、关键词和潜在问题。

```python
engine = KnowledgeEngine(
    enable_metadata_enhancement=True
)
```

**生成的元数据：**
- `summary`: 一句话摘要
- `questions`: 3-5个用户可能会问的问题
- `keywords`: 关键词列表
- `chunk_type`: 内容类型分类

**优势：**
- 提高检索准确性
- 支持多角度匹配
- 增强语义理解

## 高级检索功能

### 混合检索策略

结合向量检索和 BM25 关键词检索的优势：

```python
engine = KnowledgeEngine(
    retrieval_strategy="hybrid",
    vector_weight=0.7,      # 向量检索权重
    bm25_weight=0.3,        # BM25权重
    fusion_method="weighted" # 融合方法
)
```

**融合方法：**
- `weighted`: 加权平均（默认）
- `rrf`: 倒数排名融合（Reciprocal Rank Fusion）

### BM25 检索器

KnowledgeCore Engine 实现了完整的 BM25 算法，支持中文分词：

```python
# 纯 BM25 检索
engine = KnowledgeEngine(
    retrieval_strategy="bm25"
)
```

**特点：**
- 使用 jieba 进行中文分词
- 支持 TF-IDF 评分
- 可配置 k1 和 b 参数
- 支持元数据过滤

### 查询扩展

通过生成相关查询来提高召回率：

```python
engine = KnowledgeEngine(
    enable_query_expansion=True,
    query_expansion_method="llm",  # 或 "rule_based"
    query_expansion_count=3
)
```

**LLM 方法：**
- 使用语言模型生成语义相关的查询
- 考虑同义词和相关概念
- 适合复杂的领域特定查询

**规则方法：**
- 基于预定义的同义词词典
- 快速且成本低
- 适合常见查询模式

### 重排序 (Reranking)

使用专门的重排序模型优化检索结果：

```python
engine = KnowledgeEngine(
    enable_reranking=True,
    reranker_model="bge-reranker-v2-m3",
    reranker_provider="huggingface",
    rerank_top_k=5
)
```

**支持的模型：**
- BGE 系列：`bge-reranker-v2-m3`, `bge-reranker-large`, `bge-reranker-base`
- Cohere Rerank（需要 API 密钥）

**工作流程：**
1. 初始检索获取 top_k * 2 个候选文档
2. 使用重排序模型计算查询-文档相关性
3. 返回重排后的 top_k 个文档

## 查询优化技术

### 完整的 RAG 优化配置

```python
# 创建一个全面优化的知识引擎
engine = KnowledgeEngine(
    # 分块优化
    enable_hierarchical_chunking=True,
    enable_metadata_enhancement=True,
    chunk_size=1024,
    chunk_overlap=128,
    
    # 检索优化
    retrieval_strategy="hybrid",
    vector_weight=0.6,
    bm25_weight=0.4,
    retrieval_top_k=20,  # 初始检索更多文档
    
    # 查询优化
    enable_query_expansion=True,
    query_expansion_method="llm",
    query_expansion_count=3,
    
    # 重排序优化
    enable_reranking=True,
    reranker_model="bge-reranker-v2-m3",
    rerank_top_k=5,  # 最终返回5个最相关的
    
    # 生成优化
    temperature=0.1,
    include_citations=True,
    citation_style="inline"
)
```

### 检索流程

1. **查询扩展**：原始查询 → 多个相关查询
2. **混合检索**：
   - 向量检索：语义相似度匹配
   - BM25检索：关键词精确匹配
3. **分数融合**：使用配置的融合方法合并结果
4. **重排序**：使用交叉编码器模型精确评分
5. **生成答案**：基于最相关的文档生成回答

## 性能优化配置

### 批处理优化

```python
# 批量添加文档
result = await engine.add([
    "doc1.pdf",
    "doc2.md",
    "doc3.txt"
])

# 批量搜索
results = await engine.search_batch([
    "查询1",
    "查询2",
    "查询3"
])
```

### 缓存策略

元数据增强结果会自动缓存：
- 基于内容哈希的缓存键
- 避免重复的 LLM 调用
- 可配置缓存 TTL

### 并发控制

```python
# 在 EnhancementConfig 中配置
max_concurrent_requests=10  # 最大并发请求数
```

## 评估系统

K-Engine 提供了完整的评估框架，支持使用 Ragas 等工具对 RAG 系统进行全面评估。

### 评估指标

1. **忠实度 (Faithfulness)**：答案是否基于检索的内容
2. **答案相关性 (Answer Relevancy)**：答案是否回答了用户的问题
3. **上下文精确度 (Context Precision)**：检索的文档是否相关
4. **上下文召回率 (Context Recall)**：相关信息是否都被检索到

### 使用 Ragas 评估

```python
from knowledge_core_engine.core.evaluation import (
    create_ragas_evaluator, 
    RagasConfig,
    TestCase
)

# 配置 Ragas
config = RagasConfig(
    llm_provider="deepseek",
    embedding_provider="dashscope",
    metrics=[
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]
)

# 创建评估器
evaluator = await create_ragas_evaluator(config)

# 准备测试用例
test_case = TestCase(
    question="什么是RAG？",
    ground_truth="RAG是检索增强生成技术...",
    contexts=contexts,  # 检索到的上下文
    generated_answer=answer  # 生成的答案
)

# 运行评估
result = await evaluator.evaluate_single(test_case)
```

### 使用黄金测试集

K-Engine 提供了标准的黄金测试集用于系统评估：

```python
from examples.evaluate_with_golden_set import GoldenSetEvaluator

# 创建评估器
evaluator = GoldenSetEvaluator(
    engine=engine,
    golden_set_path="data/golden_set/rag_test_set.json"
)

# 准备测试用例
await evaluator.prepare_test_cases()

# 运行评估
report = await evaluator.evaluate(ragas_config)

# 查看结果
print(f"总体平均分: {report['overall']['mean']:.3f}")
```

### 快速评估

使用内置的快速评估脚本：

```bash
# 运行快速评估
python scripts/quick_evaluate.py
```

## 最佳实践建议

### 1. 文档类型选择

| 文档类型 | 推荐配置 |
|---------|---------|
| 技术文档 | 层级分块 + 元数据增强 + 混合检索 |
| 新闻文章 | 语义分块 + BM25检索 |
| 学术论文 | 层级分块 + 重排序 + 引用 |
| 对话记录 | 小块分割 + 纯向量检索 |

### 2. 性能与质量权衡

**高质量配置**（准确性优先）：
```python
enable_metadata_enhancement=True
enable_query_expansion=True
enable_reranking=True
retrieval_top_k=20
rerank_top_k=5
```

**高性能配置**（速度优先）：
```python
enable_metadata_enhancement=False
enable_query_expansion=False
enable_reranking=False
retrieval_top_k=5
```

**平衡配置**（推荐）：
```python
enable_metadata_enhancement=False
enable_query_expansion=True
query_expansion_method="rule_based"
enable_reranking=True
retrieval_top_k=10
rerank_top_k=5
```

### 3. 成本优化

- 使用国产模型（DeepSeek/Qwen）降低 API 成本
- 规则基础的查询扩展比 LLM 方法更经济
- 本地部署的重排序模型避免 API 调用
- 合理设置 retrieval_top_k 避免过度检索

### 4. 调试技巧

```python
# 获取详细的检索信息
result = await engine.ask(
    "你的问题",
    return_details=True
)

# 查看使用了哪些文档
for ctx in result['contexts']:
    print(f"分数: {ctx['score']}")
    print(f"内容: {ctx['content'][:100]}...")
    print(f"元数据: {ctx['metadata']}")
```

## 配置示例

### 场景1：企业知识库

```python
engine = KnowledgeEngine(
    # 保留文档结构
    enable_hierarchical_chunking=True,
    chunk_size=1024,
    
    # 混合检索确保召回率
    retrieval_strategy="hybrid",
    
    # 重排序提高准确性
    enable_reranking=True,
    
    # 包含引用增强可信度
    include_citations=True,
    citation_style="footnote"
)
```

### 场景2：客服问答系统

```python
engine = KnowledgeEngine(
    # 语义分块适合FAQ
    enable_semantic_chunking=True,
    chunk_size=256,  # 较小的块
    
    # 查询扩展处理用户表达多样性
    enable_query_expansion=True,
    query_expansion_method="llm",
    
    # 快速响应
    retrieval_top_k=5,
    enable_reranking=False
)
```

### 场景3：学术研究助手

```python
engine = KnowledgeEngine(
    # 元数据增强提取关键信息
    enable_metadata_enhancement=True,
    
    # 层级分块保留论文结构
    enable_hierarchical_chunking=True,
    
    # 高质量检索
    retrieval_strategy="hybrid",
    enable_reranking=True,
    reranker_model="bge-reranker-large",
    
    # 详细引用
    include_citations=True,
    citation_style="endnote"
)
```

## 故障排除

### 常见问题

1. **检索结果不准确**
   - 尝试启用查询扩展
   - 调整向量和BM25的权重比例
   - 启用重排序

2. **处理速度慢**
   - 减少 retrieval_top_k
   - 关闭元数据增强
   - 使用规则基础的查询扩展

3. **内存占用高**
   - 减小 chunk_size
   - 限制并发请求数
   - 定期清理缓存

4. **中文支持问题**
   - 确保安装了 jieba
   - 使用支持中文的嵌入模型
   - 选择中文优化的重排序模型

## 未来规划

- [ ] 支持更多重排序模型
- [ ] 自定义查询扩展词典
- [ ] 多语言 BM25 支持
- [ ] 增量索引更新
- [ ] 分布式检索支持

---

更多信息请参考[主文档](../README.md)或提交 [Issue](https://github.com/your-org/knowledge-core-engine/issues)。