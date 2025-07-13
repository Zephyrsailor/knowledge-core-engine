# 环境设置指南

## 重要提醒

1. **环境变量文件 (.env)**
   - **绝对不要**在 `.env` 文件的值后面添加注释
   - 错误示例：`CACHE_TTL=86400  # 这是错误的`
   - 正确示例：`CACHE_TTL=86400`
   - Pydantic 会将注释作为值的一部分，导致类型解析失败

2. **Python 环境**
   - 始终使用虚拟环境：`python -m venv .venv`
   - 激活虚拟环境后，验证 `which python` 指向 `.venv/bin/python`
   - 使用 `.venv/bin/pytest` 而不是系统的 `pytest`

3. **依赖安装**
   - 使用 `pip install -e ".[dev]"` 安装所有依赖
   - 这会安装开发依赖，包括 pytest

4. **测试运行**
   ```bash
   # 推荐方式
   .venv/bin/pytest tests/unit -v
   
   # 或者（确保虚拟环境已激活）
   python -m pytest tests/unit -v
   ```

## 常见问题排查

### 1. ModuleNotFoundError: No module named 'knowledge_core_engine'
- 确保已运行 `pip install -e .`
- 检查 `pip list | grep knowledge` 是否显示包已安装

### 2. ValidationError from Pydantic
- 检查 `.env` 文件中是否有注释
- 运行 `grep "#" .env` 查看所有注释行
- 确保数值类型的环境变量后面没有注释

### 3. 使用了错误的 Python/pytest
- 运行 `which python` 和 `which pytest`
- 应该显示 `.venv/bin/` 路径
- 如果不是，使用完整路径：`.venv/bin/pytest`

## 环境验证脚本

创建一个验证脚本来检查环境：

```bash
#!/bin/bash
echo "Python: $(which python)"
echo "Pytest: $(which pytest)"
echo "Virtual env: $VIRTUAL_ENV"
echo "Checking .env for inline comments..."
grep -E "=.*#" .env || echo "No inline comments found (good!)"
```