# 分块系统测试说明

## 测试文件结构

```
tests/unit/core/chunking/
├── __init__.py
├── test_base_chunker.py          # 基础接口测试
├── test_markdown_chunker.py      # Markdown分块器测试
├── test_smart_chunker.py         # 智能分块器测试
├── test_chunking_pipeline.py     # 分块流水线测试
└── README.md                     # 本文件
```

## 测试覆盖

### 1. 基础接口 (test_base_chunker.py)
- ✅ ChunkResult 数据类
- ✅ ChunkingResult 数据类
- ✅ BaseChunker 抽象基类
- ✅ 配置验证
- ✅ 自定义配置

### 2. Markdown分块器 (test_markdown_chunker.py)
- ✅ 按标题层级分块
- ✅ 代码块保护
- ✅ 表格保护
- ✅ 层级元数据
- ✅ 无标题文本处理
- ✅ 重叠功能
- ✅ 最小块大小
- ✅ 列表处理
- ✅ 空文档处理
- ✅ 特殊Markdown元素
- ✅ 性能测试

### 3. 智能分块器 (test_smart_chunker.py)
- ✅ 内容类型检测
- ✅ 技术文档特殊处理
- ✅ Q&A格式处理
- ✅ 上下文保留
- ✅ 实体保护
- ✅ 对话分块
- ✅ 混合内容处理
- ✅ 元数据增强

### 4. 分块流水线 (test_chunking_pipeline.py)
- ✅ 单文档处理
- ✅ 批量处理
- ✅ LlamaIndex节点创建
- ✅ 错误处理
- ✅ 空文档处理
- ✅ 元数据传播
- ✅ 性能指标
- ✅ 唯一ID生成

## 运行测试

```bash
# 运行所有分块测试
.venv/bin/pytest tests/unit/core/chunking -v

# 运行特定测试文件
.venv/bin/pytest tests/unit/core/chunking/test_base_chunker.py -v

# 运行特定测试
.venv/bin/pytest tests/unit/core/chunking/test_base_chunker.py::TestChunkResult -v
```

## 设计决策

1. **数据结构优先**：先定义清晰的数据结构（ChunkResult, ChunkingResult）
2. **抽象基类**：使用ABC确保所有分块器遵循相同接口
3. **策略模式**：不同类型的内容使用不同的分块策略
4. **元数据丰富**：每个块都携带丰富的元数据便于后续处理
5. **性能考虑**：包含性能测试确保大文档处理效率

## Mock策略

- LlamaIndex组件：完全mock，不依赖实际实现
- 文件系统：使用内存中的字符串，不创建实际文件
- 异步操作：使用AsyncMock处理异步方法

## 待实现功能

实际实现时需要考虑：
- [ ] 流式处理支持
- [ ] 自定义分块策略插件机制
- [ ] 分块质量评估
- [ ] 智能重组小块
- [ ] 多语言支持