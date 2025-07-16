# KnowledgeCore Engine 评测指南

## 概述

本文档介绍如何使用统一的评测入口 `run_evaluation.py` 进行系统评测。

## 评测架构

### 1. 双层评测体系

- **快速评测**：基于关键词覆盖率，适合开发迭代
- **专业评测**：使用Ragas框架，提供标准RAG指标

### 2. 核心评测指标

#### 2.1 Faithfulness（忠实度）
- **定义**：答案是否完全基于检索到的上下文
- **目标**：> 0.85
- **意义**：防止模型产生幻觉，确保答案的可追溯性

#### 2.2 Answer Relevancy（答案相关性）
- **定义**：答案是否真正回答了用户的问题
- **目标**：> 0.90
- **意义**：确保答案切题，不答非所问

#### 2.3 Context Precision（上下文精确度）
- **定义**：检索到的文档中有多少是真正相关的
- **目标**：> 0.80
- **意义**：评估检索系统的精确性

#### 2.4 Context Recall（上下文召回率）
- **定义**：是否检索到了回答问题所需的所有信息
- **目标**：> 0.75
- **意义**：评估知识库的完整性和检索的全面性

## 运行专业评测

### 1. 环境准备

```bash
# 安装Ragas（如果未安装）
pip install ragas

# 确保环境变量配置
export DASHSCOPE_API_KEY=your_api_key
```

### 2. 完整评测流程

```bash
# 运行完整的专业评测
python scripts/run_full_evaluation.py
```

评测流程包括：
1. 初始化优化后的RAG引擎
2. 对10个黄金测试用例生成答案（约60-90秒）
3. 运行Ragas评测（约30-60秒）
4. 生成详细报告

### 3. 评测配置

默认的专业评测配置：

```python
# 引擎配置
engine = KnowledgeEngine(
    llm_provider="qwen",
    llm_model="qwen-turbo",
    
    # 专业检索配置
    retrieval_strategy="hybrid",
    enable_query_expansion=True,
    enable_reranking=True,
    retrieval_top_k=15,
    rerank_top_k=5
)

# Ragas配置
ragas_config = RagasConfig(
    llm_provider="qwen",
    metrics=[
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ]
)
```

## 解读评测结果

### 1. 整体指标

```json
{
  "metrics": {
    "faithfulness": {
      "mean": 0.85,
      "std": 0.12
    },
    "answer_relevancy": {
      "mean": 0.92,
      "std": 0.08
    }
  }
}
```

### 2. 分类分析

- **按类别**：识别哪些知识领域表现较差
- **按难度**：了解系统在不同复杂度问题上的表现
- **按关键词**：分析答案的完整性

### 3. 性能分析

- **响应时间分布**：P50、P90、P95
- **时间瓶颈**：检索vs生成时间占比

## 优化策略

### 1. 根据Faithfulness优化

如果忠实度低于0.85：
- 检查生成模型的温度参数（建议0.1）
- 优化提示词，强调"仅基于给定上下文"
- 增加检索文档数量

### 2. 根据Answer Relevancy优化

如果相关性低于0.90：
- 改进查询理解（启用查询扩展）
- 优化检索策略（调整向量/BM25权重）
- 改进生成提示词

### 3. 根据Context Precision优化

如果精确度低于0.80：
- 启用重排序功能
- 调整检索阈值
- 优化文档分块策略

### 4. 根据Context Recall优化

如果召回率低于0.75：
- 增加retrieval_top_k
- 检查知识库完整性
- 优化查询扩展策略

## 持续改进流程

1. **建立基准线**
   ```bash
   python scripts/run_full_evaluation.py
   # 保存结果：evaluation_results/baseline_YYYYMMDD.json
   ```

2. **实施优化**
   - 根据评测结果调整配置
   - 修改代码实现

3. **验证改进**
   ```bash
   python scripts/run_full_evaluation.py
   # 对比新旧结果
   ```

4. **监控回归**
   - 在CI/CD中集成评测
   - 设置质量门槛

## 高级评测功能

### 1. 自定义测试集

```python
# 创建领域特定的测试集
golden_set = {
    "metadata": {
        "name": "医疗领域RAG测试集",
        "domain": "healthcare"
    },
    "test_cases": [
        {
            "question": "什么是高血压？",
            "ground_truth": "...",
            "expected_keywords": ["血压", "收缩压", "舒张压"]
        }
    ]
}
```

### 2. A/B测试

```python
# 对比不同配置
baseline_config = {...}
optimized_config = {...}

# 运行对比评测
results = await compare_configurations(
    baseline_config,
    optimized_config,
    test_cases
)
```

### 3. 在线评测

```python
# 实时监控系统性能
monitor = OnlineEvaluator(engine)
await monitor.start(
    sample_rate=0.1,  # 采样10%的查询
    metrics=["faithfulness", "latency"]
)
```

## 常见问题

### Q: 评测速度太慢？
A: 专业评测需要时间来保证准确性。可以：
- 使用更快的LLM（如qwen-turbo）
- 减少测试用例数量（但不建议低于5个）
- 并行运行多个评测

### Q: 指标波动较大？
A: 这可能是因为：
- LLM的随机性（设置temperature=0）
- 测试集太小（使用完整的10个用例）
- 系统不稳定（检查日志）

### Q: 如何设置合理的目标？
A: 根据业务需求：
- 金融/医疗领域：Faithfulness > 0.95
- 一般业务：Faithfulness > 0.85
- 创意领域：可适当放宽

## 总结

专业的RAG评测不仅是质量保证工具，更是持续改进的指南。通过系统化的评测和优化，可以构建高质量、可信赖的知识问答系统。