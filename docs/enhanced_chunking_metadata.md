# 增强的分块元数据设计

## 1. 层级关系元数据（支持父子级检索）

```python
{
    # 基础层级信息
    "chunk_id": "doc1_chunk_5",
    "parent_chunk_id": "doc1_chunk_2",  # 直接父块
    "root_chunk_id": "doc1_chunk_0",    # 根块（通常是文档标题）
    "child_chunk_ids": ["doc1_chunk_6", "doc1_chunk_7"],  # 子块列表
    
    # 完整路径
    "hierarchy_path": ["Introduction", "Core Concepts", "Data Structures"],
    "hierarchy_levels": [1, 2, 3],  # 对应的标题级别
    
    # 兄弟关系
    "sibling_chunks": ["doc1_chunk_4", "doc1_chunk_6"],
    "position_in_parent": 2,  # 在父节点中的位置
}
```

## 2. 上下文窗口元数据（支持多跳检索）

```python
{
    # 滑动窗口上下文
    "prev_chunks": ["doc1_chunk_3", "doc1_chunk_4"],  # 前2个块
    "next_chunks": ["doc1_chunk_6", "doc1_chunk_7"],  # 后2个块
    
    # 扩展上下文
    "context_summary": "This section discusses...",  # 周围内容摘要
    "section_summary": "The entire section covers...",  # 整节摘要
    
    # 跨块引用
    "references_to": ["doc1_chunk_12", "doc2_chunk_5"],  # 引用其他块
    "referenced_by": ["doc1_chunk_20"],  # 被其他块引用
}
```

## 3. 语义关系元数据（支持概念图谱）

```python
{
    # 概念和实体
    "key_concepts": ["RAG", "向量数据库", "嵌入"],
    "defined_terms": ["RAG"],  # 在此块中定义的术语
    "used_terms": ["向量数据库", "嵌入"],  # 使用但未定义的术语
    
    # 依赖关系
    "prerequisites": ["doc1_chunk_2"],  # 理解此块需要先读的块
    "leads_to": ["doc1_chunk_8"],  # 此块引出的后续内容
    
    # 主题标签
    "topics": ["machine_learning", "nlp", "retrieval"],
    "topic_importance": {"machine_learning": 0.8, "nlp": 0.6}
}
```

## 4. 结构化内容元数据

```python
{
    # 内容类型细分
    "content_structure": {
        "has_definition": true,
        "has_example": true,
        "has_code": true,
        "has_formula": false,
        "has_figure_reference": true
    },
    
    # 代码相关
    "code_dependencies": ["numpy", "pandas"],
    "code_functions": ["process_data", "calculate_score"],
    "code_language": "python",
    
    # 问答对
    "qa_pairs": [
        {"question": "What is RAG?", "answer_chunk_id": "doc1_chunk_6"}
    ]
}
```

## 5. 检索优化元数据

```python
{
    # 重要性评分
    "importance_score": 0.85,  # 基于位置、引用等计算
    "is_summary": false,  # 是否是摘要块
    "is_conclusion": false,  # 是否是结论块
    
    # 检索提示
    "retrieval_weight": 1.2,  # 检索时的权重调整
    "expand_context": true,  # 检索时是否需要扩展上下文
    
    # 时间相关
    "temporal_order": 5,  # 在文档中的时间顺序
    "last_modified": "2024-01-01T10:00:00Z"
}
```

## 实现示例

```python
class EnhancedChunkMetadata:
    """增强的块元数据结构"""
    
    def __init__(self):
        # 层级关系
        self.hierarchy = HierarchyMetadata()
        
        # 上下文窗口
        self.context = ContextMetadata()
        
        # 语义关系
        self.semantics = SemanticMetadata()
        
        # 结构化内容
        self.structure = StructureMetadata()
        
        # 检索优化
        self.retrieval = RetrievalMetadata()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "hierarchy": self.hierarchy.to_dict(),
            "context": self.context.to_dict(),
            "semantics": self.semantics.to_dict(),
            "structure": self.structure.to_dict(),
            "retrieval": self.retrieval.to_dict()
        }
```

## 检索场景支持

### 1. 父子级检索
```python
# 查询："RAG的实现步骤"
# 1. 找到包含"RAG实现"的块
# 2. 获取其所有子块（通过child_chunk_ids）
# 3. 如果需要背景，获取父块（通过parent_chunk_id）
```

### 2. 多跳检索
```python
# 查询："如何优化RAG性能"
# 1. 找到"RAG性能"相关块
# 2. 通过prerequisites找到前置知识块
# 3. 通过references_to找到相关技术块
# 4. 构建完整的知识路径
```

### 3. 概念图谱遍历
```python
# 查询："向量数据库在RAG中的作用"
# 1. 找到定义"向量数据库"的块
# 2. 找到同时包含"向量数据库"和"RAG"的块
# 3. 通过语义关系构建概念连接
```

### 4. 上下文扩展
```python
# 查询需要完整上下文时
# 1. 找到核心匹配块
# 2. 通过prev_chunks/next_chunks获取上下文
# 3. 根据importance_score决定扩展范围
```

## 实施建议

1. **分阶段实现**：先实现层级关系，再逐步添加其他元数据
2. **按需计算**：某些元数据（如importance_score）可以延迟计算
3. **存储优化**：使用图数据库存储复杂关系
4. **向后兼容**：保持与现有系统的兼容性

这种增强的元数据设计将大大提升系统的检索能力，支持更复杂的查询场景。