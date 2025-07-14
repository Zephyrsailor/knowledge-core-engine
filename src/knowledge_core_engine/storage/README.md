# Storage Module

该模块用于统一存储抽象。

## 计划实现的功能

- **DocumentStore** - 文档存储接口
- **MetadataStore** - 元数据存储
- **CacheStore** - 缓存管理
- **多后端支持**
  - SQLite
  - PostgreSQL
  - Redis

当前状态：基础存储功能已在 `core/embedding/vector_store.py` 中实现。
高级存储功能将在后续版本中添加。