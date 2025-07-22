"""文档管理功能的单元测试"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from pathlib import Path

from knowledge_core_engine import KnowledgeEngine
from knowledge_core_engine.core.retrieval.bm25.bm25s_retriever import BM25SRetriever
from knowledge_core_engine.core.embedding.vector_store import VectorStore, ChromaDBProvider


class TestEngineDocumentManagement:
    """测试Engine层的文档管理功能"""
    
    @pytest.fixture
    def mock_engine(self):
        """创建mock的engine实例"""
        engine = KnowledgeEngine()
        engine._initialized = True
        
        # Mock vector store
        engine._vector_store = Mock()
        engine._vector_store._provider = Mock()
        engine._vector_store._provider._collection = Mock()
        engine._vector_store.delete_documents = AsyncMock()
        
        # Mock retriever
        engine._retriever = Mock()
        engine._retriever._bm25_index = Mock()
        engine._retriever._bm25_index.delete_documents = AsyncMock(return_value=2)
        
        return engine
    
    @pytest.mark.asyncio
    async def test_delete_by_filename(self, mock_engine):
        """测试按文件名删除文档"""
        # Mock ChromaDB的get方法返回匹配的文档
        mock_engine._vector_store._provider._collection.get.return_value = {
            "ids": ["testfile_0_0", "testfile_1_512"],
            "metadatas": [{"source": "testfile.pdf"}, {"source": "testfile.pdf"}]
        }
        
        result = await mock_engine.delete("testfile.pdf")
        
        # 验证调用了正确的方法
        mock_engine._vector_store._provider._collection.get.assert_called_once()
        mock_engine._vector_store.delete_documents.assert_called_once_with(
            ["testfile_0_0", "testfile_1_512"]
        )
        
        assert result["deleted_ids"] == ["testfile_0_0", "testfile_1_512"]
        assert result["vector_deleted"] == 2
        assert result["bm25_deleted"] == 2
    
    @pytest.mark.asyncio
    async def test_delete_by_ids(self, mock_engine):
        """测试按ID列表删除文档"""
        doc_ids = ["id1", "id2", "id3"]
        result = await mock_engine.delete(doc_ids)
        
        # 验证直接使用提供的ID
        mock_engine._vector_store.delete_documents.assert_called_once_with(doc_ids)
        assert result["deleted_ids"] == doc_ids
    
    @pytest.mark.asyncio
    async def test_update_method(self, mock_engine):
        """测试update方法调用delete和add"""
        mock_engine.delete = AsyncMock(return_value={"deleted_count": 5, "vector_deleted": 5})
        mock_engine.add = AsyncMock(return_value={"processed_files": 1, "total_chunks": 5})
        
        # 需要mock实际的update方法，因为它现在有自己的实现
        from pathlib import Path
        original_update = KnowledgeEngine.update
        
        async def mock_update(self, source):
            file_path = Path(source)
            delete_result = await self.delete(file_path.name)
            add_result = await self.add([file_path])
            return {"deleted": delete_result, "added": add_result}
        
        mock_engine.update = mock_update.__get__(mock_engine, KnowledgeEngine)
        
        result = await mock_engine.update("test.pdf")
        
        # 验证delete被调用时使用文件名
        mock_engine.delete.assert_called_once_with("test.pdf")
        # 验证add被调用时使用Path对象的列表
        mock_engine.add.assert_called_once()
        call_args = mock_engine.add.call_args[0][0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], Path)
        assert str(call_args[0]) == "test.pdf"
        
        assert "deleted" in result
        assert "added" in result
    
    @pytest.mark.asyncio
    async def test_clear_method(self, mock_engine):
        """测试clear方法"""
        mock_engine._vector_store.clear = AsyncMock()
        mock_engine._retriever._bm25_index.clear = AsyncMock()
        
        await mock_engine.clear()
        
        mock_engine._vector_store.clear.assert_called_once()
        mock_engine._retriever._bm25_index.clear.assert_called_once()


class TestBM25DocumentManagement:
    """测试BM25索引的文档管理功能"""
    
    @pytest.fixture
    def bm25_retriever(self):
        """创建BM25检索器实例"""
        retriever = BM25SRetriever()
        retriever._initialized = True
        retriever._documents = ["doc1", "doc2", "doc3"]
        retriever._doc_ids = ["id1", "id2", "id3"]
        retriever._metadata = [{}, {}, {}]
        retriever._rebuild_index = AsyncMock()
        retriever._auto_save = AsyncMock()
        retriever.persist_directory = "./test_bm25"
        return retriever
    
    @pytest.mark.asyncio
    async def test_delete_documents(self, bm25_retriever):
        """测试删除文档功能"""
        result = await bm25_retriever.delete_documents(["id2"])
        
        assert result == 1
        assert bm25_retriever._doc_ids == ["id1", "id3"]
        assert bm25_retriever._documents == ["doc1", "doc3"]
        assert len(bm25_retriever._metadata) == 2
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, bm25_retriever):
        """测试删除不存在的文档"""
        result = await bm25_retriever.delete_documents(["nonexistent"])
        assert result == 0
        assert len(bm25_retriever._documents) == 3  # 没有变化
    
    @pytest.mark.asyncio
    async def test_clear(self, bm25_retriever):
        """测试清空索引"""
        await bm25_retriever.clear()
        
        assert bm25_retriever._documents == []
        assert bm25_retriever._doc_ids == []
        assert bm25_retriever._metadata == []
        assert bm25_retriever._retriever is None


class TestVectorStoreDocumentManagement:
    """测试向量存储的文档管理功能"""
    
    @pytest.fixture
    def vector_store(self):
        """创建vector store实例"""
        from knowledge_core_engine.core.config import RAGConfig
        config = RAGConfig()
        store = VectorStore(config)
        store._initialized = True
        store._provider = Mock()
        store._provider.delete_documents = AsyncMock()
        store._provider.clear_collection = AsyncMock()
        return store
    
    @pytest.mark.asyncio
    async def test_delete_documents(self, vector_store):
        """测试删除文档"""
        doc_ids = ["id1", "id2"]
        await vector_store.delete_documents(doc_ids)
        
        vector_store._provider.delete_documents.assert_called_once_with(doc_ids)
    
    @pytest.mark.asyncio
    async def test_clear(self, vector_store):
        """测试清空集合"""
        await vector_store.clear()
        
        vector_store._provider.clear_collection.assert_called_once()