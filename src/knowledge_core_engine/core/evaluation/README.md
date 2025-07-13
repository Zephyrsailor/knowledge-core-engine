# 评估模块 (Evaluation Module)

RAG系统性能评估模块，提供全面的评估指标和报告生成功能。

## 功能特性

- **多维度评估指标**
  - Faithfulness（忠实度）：评估答案是否基于提供的上下文
  - Answer Relevancy（答案相关性）：评估答案是否切题
  - Context Precision（上下文精确度）：评估检索上下文的相关性
  - Context Recall（上下文召回率）：评估相关信息的完整性

- **黄金测试集管理**
  - 支持创建、加载、保存测试集
  - 灵活的过滤和采样功能
  - 数据集拆分功能

- **多种报告格式**
  - Markdown报告（易读）
  - HTML报告（可视化）
  - JSON报告（程序化处理）

- **RAGAS框架集成**
  - 支持RAGAS评估框架
  - 提供Mock实现用于测试

## 快速开始

### 1. 基本评估

```python
import asyncio
from knowledge_core_engine.core.evaluation import Evaluator, EvaluationConfig, TestCase

async def evaluate_rag():
    # 配置评估
    config = EvaluationConfig(
        metrics=["faithfulness", "answer_relevancy"],
        llm_provider="deepseek"  # 或 "qwen", "mock"
    )
    
    # 创建评估器
    evaluator = Evaluator(config)
    
    # 创建测试用例
    test_case = TestCase(
        question="什么是RAG？",
        ground_truth="RAG是检索增强生成技术",
        contexts=["RAG结合了检索和生成..."],
        generated_answer="RAG是一种AI技术"
    )
    
    # 评估单个用例
    result = await evaluator.evaluate_single(test_case)
    print(f"平均得分: {result.average_score():.3f}")

asyncio.run(evaluate_rag())
```

### 2. 批量评估

```python
from knowledge_core_engine.core.evaluation import GoldenDataset

# 加载黄金测试集
dataset = GoldenDataset.load("data/golden_set/rag_qa_dataset.json")

# 为每个测试用例生成答案
for test_case in dataset:
    test_case.generated_answer = your_rag_system(test_case.question)

# 批量评估
results = await evaluator.evaluate_batch(list(dataset))

# 生成摘要
summary = evaluator.generate_summary(results)
print(f"整体得分: {summary['overall_score']:.3f}")
```

### 3. 生成评估报告

```python
from knowledge_core_engine.core.evaluation.report_generator import ReportGenerator

# 创建报告生成器
report_gen = ReportGenerator()

# 生成Markdown报告
report = report_gen.generate_report(
    results,
    output_path="reports/evaluation_report.md",
    format="markdown",
    metadata={"dataset": "test_set", "date": "2024-01-15"}
)

# 生成HTML报告（带可视化）
html_report = report_gen.generate_report(
    results,
    output_path="reports/evaluation_report.html",
    format="html"
)
```

## 评估指标详解

### Faithfulness（忠实度）
- **定义**：衡量生成答案中的陈述是否可以从提供的上下文中推断出来
- **计算方法**：
  1. 从答案中提取所有事实性陈述
  2. 验证每个陈述是否被上下文支持
  3. 得分 = 支持的陈述数 / 总陈述数

### Answer Relevancy（答案相关性）
- **定义**：衡量答案与问题的相关程度
- **计算方法**：
  1. 从答案生成可能的问题
  2. 计算生成问题与原问题的相似度
  3. 取平均相似度作为得分

### Context Precision（上下文精确度）
- **定义**：检索到的上下文中相关内容的比例
- **计算方法**：
  1. 评估每个上下文片段的相关性
  2. 得分 = 相关上下文数 / 总上下文数

### Context Recall（上下文召回率）
- **定义**：ground truth中的信息被上下文覆盖的程度
- **计算方法**：
  1. 从ground truth提取关键信息点
  2. 检查每个信息点是否被上下文覆盖
  3. 得分 = 覆盖的信息点数 / 总信息点数

## 黄金测试集格式

```json
{
  "metadata": {
    "name": "dataset_name",
    "version": "1.0.0",
    "description": "数据集描述"
  },
  "test_cases": [
    {
      "test_case_id": "unique_id",
      "question": "问题文本",
      "ground_truth": "标准答案",
      "contexts": ["相关上下文1", "相关上下文2"],
      "metadata": {
        "category": "分类",
        "difficulty": "难度"
      }
    }
  ]
}
```

## 自定义评估指标

```python
from knowledge_core_engine.core.evaluation import MetricResult

# 定义自定义指标
async def custom_length_metric(answer: str, **kwargs) -> MetricResult:
    """评估答案长度的合理性"""
    length = len(answer)
    
    # 自定义评分逻辑
    if 50 <= length <= 200:
        score = 1.0
    elif length < 50:
        score = length / 50
    else:
        score = max(0.5, 1.0 - (length - 200) / 1000)
    
    return MetricResult(
        name="answer_length",
        score=score,
        details={"length": length}
    )

# 注册自定义指标
evaluator.register_metric("answer_length", custom_length_metric)
```

## 配置选项

```python
config = EvaluationConfig(
    # 评估指标
    metrics=["faithfulness", "answer_relevancy"],
    
    # LLM配置
    llm_provider="deepseek",  # deepseek, qwen, openai, mock
    llm_model="deepseek-chat",
    temperature=0.0,  # 评估使用确定性输出
    
    # 处理配置
    batch_size=10,
    max_concurrent_evaluations=5,
    
    # 缓存配置
    use_cache=True,
    cache_dir=Path("~/.cache/k-engine/evaluation"),
    
    # 输出配置
    output_format="json",  # json, csv, markdown
    verbose=True
)
```

## 性能优化建议

1. **批处理**：使用`evaluate_batch`而不是多次调用`evaluate_single`
2. **缓存**：启用缓存避免重复计算
3. **并发控制**：调整`max_concurrent_evaluations`平衡速度和资源
4. **选择性评估**：只选择必要的指标，避免过度评估

## 故障排除

### 常见问题

1. **LLM调用失败**
   - 检查API密钥配置
   - 使用mock provider进行测试
   - 检查网络连接

2. **评估速度慢**
   - 减少并发数
   - 启用缓存
   - 使用更快的LLM provider

3. **内存不足**
   - 减小batch_size
   - 分批处理大数据集

## 示例脚本

完整示例请参考 `examples/evaluation_example.py`