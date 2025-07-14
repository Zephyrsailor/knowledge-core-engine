# Pipelines Module

该模块用于组织高级工作流程。

## 计划实现的功能

- **IngestionPipeline** - 文档摄入流水线
- **RetrievalPipeline** - 检索流水线
- **GenerationPipeline** - 生成流水线
- **EvaluationPipeline** - 评估流水线

当前状态：功能已整合到 `KnowledgeEngine` 类中，提供更简单的API。
如需底层控制，请使用 `core` 模块中的组件。