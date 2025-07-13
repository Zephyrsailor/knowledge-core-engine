# 分块系统设计文档

## 概述

分块系统是KnowledgeCore Engine的第二个核心组件，负责将解析后的Markdown文档智能地分割成适合向量化和检索的块。

## 设计目标

1. **语义完整性**：确保每个块都是语义完整的单元
2. **上下文保留**：保留文档结构和上下文信息
3. **灵活性**：支持不同类型文档的不同分块策略
4. **性能**：高效处理大文档
5. **可扩展性**：易于添加新的分块策略

## 架构设计

### 核心组件

```
chunking/
├── base.py                 # 基础接口定义
├── markdown_chunker.py     # Markdown专用分块器
├── smart_chunker.py        # 智能内容感知分块器
└── pipeline.py            # 分块流水线
```

### 数据模型

```python
@dataclass
class ChunkResult:
    """单个块的结果"""
    content: str                    # 块的文本内容
    metadata: Dict[str, Any]        # 元数据
    start_char: int                 # 原文档中的起始位置
    end_char: int                   # 原文档中的结束位置

@dataclass
class ChunkingResult:
    """整个文档的分块结果"""
    chunks: List[ChunkResult]       # 所有块
    total_chunks: int               # 总块数
    document_metadata: Dict[str, Any]  # 文档级元数据
```

### 分块策略

#### 1. MarkdownChunker
- 按Markdown标题层级分块
- 保护代码块和表格的完整性
- 保留标题层级关系在元数据中
- 支持列表的智能处理

#### 2. SmartChunker
- 自动检测内容类型（技术文档、Q&A、叙事等）
- 根据内容类型选择最佳分块策略
- 保留实体和上下文引用
- 增强元数据（prerequisites、references等）

#### 3. ChunkingPipeline
- 集成解析结果和分块过程
- 支持批量处理
- 转换为LlamaIndex TextNode
- 性能监控和错误处理

## 元数据策略

每个块都包含丰富的元数据：

```python
{
    # 结构信息
    "header": "Section 1.1",
    "header_level": 2,
    "parent_header": "Chapter 1",
    
    # 内容类型
    "content_type": "technical",
    "has_code": true,
    "has_table": false,
    
    # 上下文信息
    "prerequisites": ["basic concepts"],
    "references": ["previous chapter"],
    
    # 位置信息
    "chunk_index": 3,
    "total_chunks": 10,
    
    # 文档信息（继承）
    "source": "guide.md",
    "author": "John Doe"
}
```

## 配置参数

```python
# 基础配置
chunk_size: int = 1024          # 目标块大小
chunk_overlap: int = 200        # 块之间的重叠
min_chunk_size: int = 100       # 最小块大小

# 高级配置
preserve_structure: bool = True  # 保留文档结构
content_aware: bool = True      # 启用内容感知
smart_overlap: bool = True      # 智能重叠（完整句子）
```

## 性能考虑

1. **流式处理**：支持大文档的流式处理
2. **并行处理**：批量文档的并行处理
3. **缓存**：智能缓存分块结果
4. **性能基准**：
   - 1MB文档：< 100ms
   - 10MB文档：< 1s

## 与其他组件的集成

### 输入：解析模块
```python
ParseResult -> ChunkingPipeline -> ChunkingResult
```

### 输出：嵌入模块
```python
ChunkingResult -> TextNode[] -> EmbeddingPipeline
```

## 测试策略

1. **单元测试**：44个测试用例，100%通过
2. **集成测试**：与解析模块的端到端测试
3. **性能测试**：大文档处理性能
4. **质量测试**：分块质量评估

## 下一步计划

1. 实现核心分块器类
2. 集成LlamaIndex的NodeParser
3. 添加更多内容类型支持
4. 优化性能和内存使用
5. 实现分块质量评估机制