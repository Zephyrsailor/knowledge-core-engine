# RAGAS评测系统设计

## 评测流程

### 方案A：理想评测（使用预定义contexts）
适用于：快速评估生成质量，不考虑检索效果

```
1. 加载测试数据集（含预定义contexts）
2. 使用预定义contexts生成答案
3. 评估生成质量（Faithfulness, Answer Relevancy）
```

### 方案B：端到端评测（真实RAG流程）
适用于：评估完整系统性能

```
1. 准备知识库文档
2. 构建向量数据库
3. 对每个测试问题：
   - 从知识库检索contexts
   - 基于检索结果生成答案
   - 与预定义contexts对比（评估检索质量）
   - 与ground_truth对比（评估生成质量）
4. 计算RAGAS指标
```

## 数据准备策略

### 1. 构建评测知识库
需要创建包含测试集中所有contexts内容的文档：

```python
# 从测试集提取所有contexts，构建源文档
def create_evaluation_docs(test_dataset):
    docs = []
    for test_case in test_dataset:
        # 将contexts组合成文档
        doc_content = "\n\n".join(test_case["contexts"])
        docs.append({
            "id": test_case["test_case_id"],
            "content": doc_content,
            "metadata": test_case["metadata"]
        })
    return docs
```

### 2. 改进测试数据集结构
添加源文档引用：

```json
{
  "test_case_id": "rag_001",
  "question": "什么是RAG技术？",
  "source_docs": ["rag_intro.md", "rag_architecture.md"],  // 新增
  "expected_contexts": [...],  // 原contexts改名
  "ground_truth": "..."
}
```

## RAGAS指标解释

### 1. Faithfulness（忠实度）
- 评估答案是否完全基于提供的上下文
- 防止模型"编造"信息

### 2. Answer Relevancy（答案相关性）
- 评估答案是否切题
- 确保回答了用户的问题

### 3. Context Precision（上下文精确度）
- 评估检索到的上下文中相关内容的比例
- 检索系统的精确性

### 4. Context Recall（上下文召回率）
- 评估是否检索到所有相关信息
- 检索系统的完整性

## 实施建议

### 短期方案（快速验证）
1. 使用现有golden_qa_dataset.json
2. 创建一个包含所有contexts的合成文档
3. 将此文档加入知识库
4. 运行端到端评测

### 长期方案（生产级评测）
1. 准备真实的源文档集
2. 在测试数据中标注源文档
3. 实现检索结果与期望contexts的对比
4. 建立持续评测机制