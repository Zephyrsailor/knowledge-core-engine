"""
真正的端到端测试 - 不使用任何mock

这些测试验证整个系统的功能，从文档添加到查询生成
"""

import pytest
import asyncio
import os
from pathlib import Path
import tempfile
import shutil
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_core_engine import KnowledgeEngine


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_engine_basic_flow():
    """测试引擎的基本流程 - 添加文档并查询"""
    # 设置测试环境变量
    os.environ.setdefault("DEEPSEEK_API_KEY", "test_key")
    os.environ.setdefault("DASHSCOPE_API_KEY", "test_key")
    
    # 创建测试目录
    test_dir = Path(tempfile.mkdtemp()) / "test_docs"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试文档
    test_file = test_dir / "test_knowledge.md"
    test_file.write_text("""
# 测试知识库

## 什么是RAG？
RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的AI技术。

## 核心组件
1. 检索器：从知识库中查找相关信息
2. 生成器：基于检索到的信息生成答案
3. 知识库：存储结构化的文档和信息
""")
    
    try:
        # 创建引擎
        engine = KnowledgeEngine(
            persist_directory=str(test_dir / "vector_db")
        )
        
        # 添加文档
        result = await engine.add(str(test_file))
        print(f"添加结果: {result}")
        if result["failed_files"]:
            print(f"失败文件: {result['failed_files']}")
        assert result["total_files"] == 1
        assert result["processed_files"] == 1
        assert result["total_chunks"] > 0
        assert len(result["failed_files"]) == 0
        
        # 查询
        answer = await engine.ask("什么是RAG?")
        assert answer is not None
        assert len(answer) > 0
        assert "抱歉" not in answer  # 应该找到答案
        
        # 搜索
        search_results = await engine.search("检索器", top_k=3)
        assert len(search_results) > 0
        assert search_results[0]["content"] is not None
        
        # 详细查询
        detailed = await engine.ask_with_details("核心组件有哪些？")
        assert detailed["answer"] is not None
        assert len(detailed["contexts"]) > 0
        assert "question" in detailed
        
        # 关闭引擎
        await engine.close()
        
    finally:
        # 清理
        shutil.rmtree(test_dir.parent)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_engine_multiple_files():
    """测试处理多个文件"""
    os.environ.setdefault("DEEPSEEK_API_KEY", "test_key")
    os.environ.setdefault("DASHSCOPE_API_KEY", "test_key")
    
    test_dir = Path(tempfile.mkdtemp()) / "test_docs"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建多个测试文档
    files = []
    
    # 文档1
    file1 = test_dir / "doc1.md"
    file1.write_text("# 文档1\n\n这是第一个测试文档。")
    files.append(file1)
    
    # 文档2
    file2 = test_dir / "doc2.txt"
    file2.write_text("文档2内容：这是第二个测试文档。")
    files.append(file2)
    
    try:
        engine = KnowledgeEngine()
        
        # 添加多个文件
        result = await engine.add([str(f) for f in files])
        assert result["total_files"] == 2
        assert result["processed_files"] == 2
        
        await engine.close()
        
    finally:
        shutil.rmtree(test_dir.parent)


@pytest.mark.e2e  
@pytest.mark.asyncio
async def test_engine_error_handling():
    """测试错误处理"""
    os.environ.setdefault("DEEPSEEK_API_KEY", "test_key")
    os.environ.setdefault("DASHSCOPE_API_KEY", "test_key")
    
    # 使用临时目录避免干扰
    test_dir = Path(tempfile.mkdtemp()) / "error_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        engine = KnowledgeEngine(
            persist_directory=str(test_dir / "vector_db")
        )
        
        # 测试不存在的文件
        result = await engine.add("non_existent_file.pdf")
        assert result["total_files"] == 1
        assert result["processed_files"] == 0
        assert len(result["failed_files"]) == 1
        
        # 测试空知识库查询
        answer = await engine.ask("测试问题")
        assert "抱歉" in answer or "没有找到" in answer
        
        await engine.close()
        
    finally:
        # 清理
        shutil.rmtree(test_dir.parent)


if __name__ == "__main__":
    # 可以直接运行测试
    asyncio.run(test_engine_basic_flow())
    print("✅ 基本流程测试通过")
    
    asyncio.run(test_engine_multiple_files()) 
    print("✅ 多文件测试通过")
    
    asyncio.run(test_engine_error_handling())
    print("✅ 错误处理测试通过")