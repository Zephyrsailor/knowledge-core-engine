# KnowledgeCore Engine 文档管理指南

## 概述

KnowledgeCore Engine 提供了完整的文档管理功能，包括添加、列出、删除和更新文档。本文档说明了文档管理的设计理念和使用方法。

## 核心功能

### 1. 添加文档 (Add)
```python
result = await engine.add(["path/to/document.pdf"])
```

### 2. 列出文档 (List)
```python
# 列出所有文档
doc_list = await engine.list()

# 带过滤条件
doc_list = await engine.list(
    filter={"file_type": "pdf"},
    page=1,
    page_size=20,
    return_stats=True
)
```

**返回格式：**
```json
{
    "documents": [
        {
            "name": "文档名.pdf",
            "path": "/path/to/文档名.pdf",
            "chunks_count": 10,
            "total_size": 1024,
            "created_at": "2024-01-01T00:00:00",
            "metadata": {...}
        }
    ],
    "total": 100,
    "page": 1,
    "page_size": 20,
    "pages": 5
}
```

### 3. 删除文档 (Delete)
```python
# 按文件名删除
result = await engine.delete(source="document.pdf")

# 按文档ID删除
result = await engine.delete(doc_ids=["doc_id_1", "doc_id_2"])
```

### 4. 更新文档 (Update)
```python
# 更新文档（删除后重新添加）
result = await engine.update("path/to/updated_document.pdf")
```

## 文档存储架构

### 向量索引管理
- **存储位置**: ChromaDB（默认在 `data/chroma_db/`）
- **存储内容**: 文档内容、嵌入向量、元数据
- **文档ID格式**: `{文件名}_{chunk_index}_{start_char}`

### 原始文件管理策略

KnowledgeCore Engine 采用**索引与文件分离**的策略：

1. **引擎不管理原始文件**
   - 原始文件保留在用户指定的位置
   - 引擎只维护文档的向量索引
   - 删除操作只删除索引，不删除原始文件

2. **为什么这样设计？**
   - 避免重复存储，节省空间
   - 用户保持对原始文件的完全控制
   - 适应多种部署场景（本地、云端、分布式）
   - 简化权限管理

3. **最佳实践**
   - 建立专门的文档目录（如 `data/source_docs/`）
   - 使用版本控制管理文档变更
   - 定期备份原始文件和向量索引

## API 接口

### REST API 端点

#### 列出文档
```http
GET /documents?page=1&page_size=20&file_type=pdf&return_stats=true
```

**查询参数：**
- `page`: 页码（默认1）
- `page_size`: 每页数量（默认20，最大100）
- `file_type`: 文件类型过滤
- `name_pattern`: 文件名模式匹配
- `return_stats`: 是否返回统计信息

#### 删除文档
```http
DELETE /documents/{document_name}
```

## 使用示例

### Python 代码示例
```python
from knowledge_core_engine import KnowledgeEngine, KnowledgeConfig

# 初始化引擎
config = KnowledgeConfig()
engine = KnowledgeEngine(config)

# 添加文档
await engine.add(["docs/manual.pdf", "docs/guide.md"])

# 列出所有PDF文档
pdf_docs = await engine.list(filter={"file_type": "pdf"})
for doc in pdf_docs["documents"]:
    print(f"名称: {doc['name']}, 分块数: {doc['chunks_count']}")

# 删除特定文档
await engine.delete(source="old_manual.pdf")

# 关闭引擎
await engine.close()
```

### 命令行示例
```bash
# 运行API服务器
python examples/api_server_simple.py

# 列出文档
curl http://localhost:8000/documents

# 带过滤的列表
curl "http://localhost:8000/documents?file_type=pdf&page_size=10"

# 删除文档
curl -X DELETE http://localhost:8000/documents/manual.pdf
```

## 注意事项

1. **文档去重**：系统会自动检测重复文档，避免重复索引
2. **大文件处理**：大文件会自动分块处理，每个块独立索引
3. **元数据增强**：启用后会为每个文档块生成摘要、问题等元数据
4. **性能优化**：使用分页避免一次加载过多文档

## 故障排除

### 常见问题

1. **列表返回空**
   - 检查向量数据库是否正确初始化
   - 确认文档已成功添加

2. **删除失败**
   - 检查文档名是否正确
   - 确认文档存在于索引中

3. **过滤不生效**
   - 确认过滤条件格式正确
   - 检查元数据中是否包含相应字段

## 未来增强

- [ ] 支持批量操作API
- [ ] 添加文档版本管理
- [ ] 实现增量更新（只更新变化的部分）
- [ ] 支持更复杂的查询条件
- [ ] 添加文档标签系统