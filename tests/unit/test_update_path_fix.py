"""测试update方法的路径处理修复"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from knowledge_core_engine.engine import KnowledgeEngine


class TestUpdatePathFix:
    """测试update方法对不同路径格式的处理"""

    @pytest.fixture
    async def engine(self):
        """创建测试引擎实例"""
        engine = KnowledgeEngine(
            llm_provider="deepseek",
            embedding_provider="dashscope"
        )
        
        # Mock 依赖
        engine._vector_store = Mock()
        engine._retriever = Mock()
        engine._parser = Mock()
        engine._chunker = Mock()
        engine._embedder = Mock()
        engine._metadata_enhancer = None
        engine._initialized = True
        
        # Mock vector store provider
        engine._vector_store._provider = Mock()
        engine._vector_store._provider._collection = Mock()
        
        return engine

    @pytest.mark.asyncio
    async def test_update_with_relative_path(self, engine):
        """测试使用相对路径更新文档"""
        # 设置mock数据
        engine._vector_store._provider._collection.get.return_value = {
            "ids": ["中华人民共和国婚姻法_0_0", "中华人民共和国婚姻法_1_512"],
            "documents": ["content1", "content2"],
            "metadatas": [{"source": "中华人民共和国婚姻法.pdf"}, {"source": "中华人民共和国婚姻法.pdf"}]
        }
        
        # Mock delete操作
        engine._vector_store.delete_documents = AsyncMock()
        
        # Mock add操作返回值
        mock_parse_result = Mock()
        mock_parse_result.content = "test content"
        mock_parse_result.metadata = {"parse_method": "test"}
        engine._parser.process = AsyncMock(return_value=mock_parse_result)
        
        mock_chunk_result = Mock()
        mock_chunk_result.chunks = []
        mock_chunk_result.total_chunks = 0
        engine._chunker.process_parse_result = AsyncMock(return_value=mock_chunk_result)
        
        # 执行update - 使用相对路径
        result = await engine.update("data/source_docs/中华人民共和国婚姻法.pdf")
        
        # 验证delete被调用时使用的是文件名，而不是完整路径
        # delete方法应该使用文件名 "中华人民共和国婚姻法.pdf"
        assert engine._vector_store.delete_documents.called
        
        # 验证结果
        assert "deleted" in result
        assert "added" in result
        assert result["deleted"]["deleted_ids"] == ["中华人民共和国婚姻法_0_0", "中华人民共和国婚姻法_1_512"]

    @pytest.mark.asyncio
    async def test_update_with_absolute_path(self, engine):
        """测试使用绝对路径更新文档"""
        # 设置mock数据
        engine._vector_store._provider._collection.get.return_value = {
            "ids": ["test_document_0_0", "test_document_1_512"],
            "documents": ["content1", "content2"],
            "metadatas": [{"source": "test_document.pdf"}, {"source": "test_document.pdf"}]
        }
        
        # Mock delete操作
        engine._vector_store.delete_documents = AsyncMock()
        
        # Mock add操作
        mock_parse_result = Mock()
        mock_parse_result.content = "test content"
        mock_parse_result.metadata = {"parse_method": "test"}
        engine._parser.process = AsyncMock(return_value=mock_parse_result)
        
        mock_chunk_result = Mock()
        mock_chunk_result.chunks = []
        mock_chunk_result.total_chunks = 0
        engine._chunker.process_parse_result = AsyncMock(return_value=mock_chunk_result)
        
        # 执行update - 使用绝对路径
        result = await engine.update("/absolute/path/to/test_document.pdf")
        
        # 验证结果
        assert "deleted" in result
        assert "added" in result
        assert result["deleted"]["deleted_ids"] == ["test_document_0_0", "test_document_1_512"]

    @pytest.mark.asyncio
    async def test_update_with_filename_only(self, engine):
        """测试仅使用文件名更新文档"""
        # 设置mock数据
        engine._vector_store._provider._collection.get.return_value = {
            "ids": ["simple_0_0"],
            "documents": ["content"],
            "metadatas": [{"source": "simple.md"}]
        }
        
        # Mock delete操作
        engine._vector_store.delete_documents = AsyncMock()
        
        # Mock add操作
        mock_parse_result = Mock()
        mock_parse_result.content = "test content"
        mock_parse_result.metadata = {"parse_method": "test"}
        engine._parser.process = AsyncMock(return_value=mock_parse_result)
        
        mock_chunk_result = Mock()
        mock_chunk_result.chunks = []
        mock_chunk_result.total_chunks = 0
        engine._chunker.process_parse_result = AsyncMock(return_value=mock_chunk_result)
        
        # 执行update - 仅文件名
        result = await engine.update("simple.md")
        
        # 验证结果
        assert "deleted" in result
        assert "added" in result
        assert result["deleted"]["deleted_ids"] == ["simple_0_0"]

    @pytest.mark.asyncio
    async def test_delete_with_different_inputs(self, engine):
        """测试delete方法对不同输入格式的处理"""
        # 设置mock数据
        engine._vector_store._provider._collection.get.return_value = {
            "ids": ["test_0_0", "test_1_512", "other_0_0"],
            "documents": ["content1", "content2", "content3"],
            "metadatas": [
                {"source": "test.pdf"}, 
                {"source": "test.pdf"},
                {"source": "other.txt"}
            ]
        }
        
        engine._vector_store.delete_documents = AsyncMock()
        
        # 测试1：使用文件名删除
        result = await engine.delete("test.pdf")
        assert len(result["deleted_ids"]) == 2
        assert "test_0_0" in result["deleted_ids"]
        assert "test_1_512" in result["deleted_ids"]
        
        # 测试2：使用路径删除（应该提取文件名）
        result = await engine.delete("path/to/test.pdf")
        assert len(result["deleted_ids"]) == 2
        
        # 测试3：使用文档ID列表删除
        result = await engine.delete(["test_0_0", "other_0_0"])
        assert result["deleted_ids"] == ["test_0_0", "other_0_0"]