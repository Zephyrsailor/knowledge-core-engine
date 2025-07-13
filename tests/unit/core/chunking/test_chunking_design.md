# 分块系统测试设计

## 测试覆盖范围

### 1. 基础分块器接口测试 (test_base_chunker.py)
- [ ] BaseChunker是抽象基类
- [ ] 必须实现chunk方法
- [ ] ChunkResult数据类包含正确的字段
- [ ] 支持的参数配置

### 2. Markdown分块器测试 (test_markdown_chunker.py)
- [ ] 按标题层级分块
- [ ] 保留层级关系在元数据中
- [ ] 处理代码块（不拆分代码块）
- [ ] 处理表格（不拆分表格）
- [ ] 处理列表（智能处理列表项）
- [ ] 尊重最小/最大块大小限制
- [ ] 处理空文档
- [ ] 处理无标题的纯文本

### 3. 智能分块器测试 (test_smart_chunker.py)
- [ ] 根据文档类型选择策略
- [ ] 技术文档的特殊处理
- [ ] 对话/问答格式的处理
- [ ] 保留上下文信息
- [ ] 重叠（overlap）功能
- [ ] 元数据增强（章节信息、位置信息等）

### 4. LlamaIndex集成测试 (test_llama_index_chunker.py)
- [ ] 与LlamaIndex NodeParser的集成
- [ ] TextNode的正确创建
- [ ] 元数据的正确传递
- [ ] 层级节点解析

### 5. 分块管道测试 (test_chunking_pipeline.py)
- [ ] 端到端工作流：ParseResult -> Chunks
- [ ] 批处理多个文档
- [ ] 错误处理和恢复
- [ ] 性能测试（大文档）

## 测试数据结构

```python
@dataclass
class ChunkResult:
    """单个块的结果"""
    content: str  # 块的文本内容
    metadata: Dict[str, Any]  # 元数据
    start_char: int  # 在原文档中的起始位置
    end_char: int  # 在原文档中的结束位置
    
@dataclass
class ChunkingResult:
    """整个文档的分块结果"""
    chunks: List[ChunkResult]
    total_chunks: int
    document_metadata: Dict[str, Any]  # 文档级别的元数据
```

## 关键测试场景

### 场景1：技术文档分块
输入：包含多级标题、代码块、表格的Markdown
期望：
- 按逻辑结构分块
- 代码块保持完整
- 保留标题层级信息

### 场景2：长文本分块
输入：超过chunk_size的连续文本
期望：
- 在句子边界分割
- 保持overlap
- 不超过max_chunk_size

### 场景3：混合内容分块
输入：标题+文本+代码+列表的组合
期望：
- 智能识别内容类型
- 适当的分块策略
- 元数据标注内容类型

## Mock策略

1. **LlamaIndex组件**：mock掉实际的NodeParser
2. **配置加载**：使用测试配置而非实际配置文件
3. **性能测试**：生成synthetic data而非真实文档

## 性能基准

- 1MB Markdown文档：< 100ms
- 10MB Markdown文档：< 1s
- 支持流式处理大文档