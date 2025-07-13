# RAG System Evaluation Report

**Generated**: 2025-07-13 11:54:35

## Configuration
- **dataset**: rag_qa_dataset
- **model**: mock
- **date**: 2024-01-15

## Summary Statistics

- **Total Test Cases**: 5
- **Overall Score**: 0.850

### Metric Averages
- **faithfulness**: 0.850
- **answer_relevancy**: 0.900
- **context_precision**: 0.800

### Score Distribution

| Metric | Min | Max | Mean | Std Dev |
|--------|-----|-----|------|---------|
| faithfulness | 0.850 | 0.850 | 0.850 | 0.000 |
| answer_relevancy | 0.900 | 0.900 | 0.900 | 0.000 |
| context_precision | 0.800 | 0.800 | 0.800 | 0.000 |

## Detailed Results

### Test Case 1: rag_001

**Generated Answer**: RAG is a technique that combines retrieval and generation for better AI responses....

**Scores**:
- faithfulness: 0.850
- answer_relevancy: 0.900
- context_precision: 0.800

**Metadata**:
- category: definition
- difficulty: easy
- domain: 技术概念

### Test Case 2: rag_002

**Generated Answer**: RAG的核心优势包括：知识可更新性：无需重新训练模型即可更新知识库 可解释性：每个答案都可追溯到具体的源文档......

**Scores**:
- faithfulness: 0.850
- answer_relevancy: 0.900
- context_precision: 0.800

**Metadata**:
- category: advantages
- difficulty: medium
- domain: 技术特性

### Test Case 3: rag_003

**Generated Answer**: 实施RAG系统的关键步骤：第一步：文档准备和预处理，包括格式转换、清洗和标准化 第二步：文档分块策略设计，需要考虑语义完整性和检索效率的平衡......

**Scores**:
- faithfulness: 0.850
- answer_relevancy: 0.900
- context_precision: 0.800

**Metadata**:
- category: implementation
- difficulty: hard
- domain: 实施指南

### Test Case 4: rag_004

**Generated Answer**: 向量数据库用于存储文档的向量表示，支持高效的相似度搜索 向量数据库能够快速找到与查询向量最相似的文档片段......

**Scores**:
- faithfulness: 0.850
- answer_relevancy: 0.900
- context_precision: 0.800

**Metadata**:
- category: component
- difficulty: medium
- domain: 系统组件

### Test Case 5: rag_005

**Generated Answer**: 文档分块（Chunking）是将长文档分割成较小片段的过程 分块重要的原因：嵌入模型通常有输入长度限制（如512或1024个token）......

**Scores**:
- faithfulness: 0.850
- answer_relevancy: 0.900
- context_precision: 0.800

**Metadata**:
- category: concept
- difficulty: medium
- domain: 技术概念

## Performance Insights
- ✅ Excellent overall performance (>85%)