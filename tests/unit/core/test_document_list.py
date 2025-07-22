"""测试文档列表功能"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from knowledge_core_engine.engine import KnowledgeEngine


class TestDocumentList:
    """测试文档列表功能"""

    @pytest.fixture
    async def engine(self):
        """创建测试引擎实例"""
        engine = KnowledgeEngine(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            enable_metadata_enhancement=False,
            retrieval_strategy="hybrid"
        )
        
        # Mock 依赖
        engine._vector_store = Mock()
        engine._retriever = Mock()
        engine._initialized = True
        
        return engine

    @pytest.mark.asyncio
    async def test_list_empty_knowledge_base(self, engine):
        """测试空知识库的list"""
        # Mock空的返回
        engine._vector_store.list_documents = AsyncMock(return_value={
            "documents": [],
            "total": 0,
            "page": 1,
            "page_size": 20,
            "pages": 0
        })
        
        result = await engine.list()
        
        assert result["documents"] == []
        assert result["total"] == 0
        assert result["page"] == 1
        assert result["page_size"] == 20
        assert result["pages"] == 0

    @pytest.mark.asyncio
    async def test_list_with_documents(self, engine):
        """测试有文档的list"""
        # Mock文档数据
        mock_docs = {
            "documents": [
                {
                    "name": "test1.pdf",
                    "path": "/path/to/test1.pdf",
                    "chunks_count": 5,
                    "total_size": 1024,
                    "created_at": "2024-01-01T00:00:00",
                    "metadata": {"source": "test1.pdf"}
                },
                {
                    "name": "test2.md",
                    "path": "/path/to/test2.md",
                    "chunks_count": 3,
                    "total_size": 512,
                    "created_at": "2024-01-02T00:00:00",
                    "metadata": {"source": "test2.md"}
                }
            ],
            "total": 2,
            "page": 1,
            "page_size": 20,
            "pages": 1
        }
        engine._vector_store.list_documents = AsyncMock(return_value=mock_docs)
        
        result = await engine.list()
        
        assert len(result["documents"]) == 2
        assert result["total"] == 2
        assert result["documents"][0]["name"] == "test1.pdf"
        assert result["documents"][1]["name"] == "test2.md"

    @pytest.mark.asyncio
    async def test_list_with_filter(self, engine):
        """测试带过滤条件的list"""
        # Mock只返回PDF文档
        mock_docs = {
            "documents": [
                {
                    "name": "test1.pdf",
                    "path": "/path/to/test1.pdf",
                    "chunks_count": 5,
                    "total_size": 1024,
                    "created_at": "2024-01-01T00:00:00",
                    "metadata": {"source": "test1.pdf"}
                }
            ],
            "total": 1,
            "page": 1,
            "page_size": 20,
            "pages": 1
        }
        engine._vector_store.list_documents = AsyncMock(return_value=mock_docs)
        
        result = await engine.list(filter={"file_type": "pdf"})
        
        assert len(result["documents"]) == 1
        assert result["documents"][0]["name"].endswith(".pdf")

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, engine):
        """测试分页功能"""
        # Mock大量文档
        mock_docs = []
        for i in range(50):
            mock_docs.append({
                "name": f"test{i}.pdf",
                "path": f"/path/to/test{i}.pdf",
                "chunks_count": 5,
                "total_size": 1024,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"source": f"test{i}.pdf"}
            })
        
        # 模拟分页返回
        async def mock_list_with_pagination(filter=None, page=1, page_size=20, **kwargs):
            start = (page - 1) * page_size
            end = start + page_size
            total_pages = (50 + page_size - 1) // page_size
            return {
                "documents": mock_docs[start:end],
                "total": 50,
                "page": page,
                "page_size": page_size,
                "pages": total_pages
            }
        
        engine._vector_store.list_documents = mock_list_with_pagination
        engine._vector_store.count_documents = AsyncMock(return_value=50)
        
        # 测试第一页
        result = await engine.list(page=1, page_size=20)
        assert len(result["documents"]) == 20
        assert result["total"] == 50
        assert result["page"] == 1
        assert result["pages"] == 3
        
        # 测试第二页
        result = await engine.list(page=2, page_size=20)
        assert len(result["documents"]) == 20
        assert result["page"] == 2
        
        # 测试最后一页
        result = await engine.list(page=3, page_size=20)
        assert len(result["documents"]) == 10
        assert result["page"] == 3

    @pytest.mark.asyncio
    async def test_list_without_stats(self, engine):
        """测试不返回统计信息"""
        mock_docs = {
            "documents": [
                {
                    "name": "test1.pdf",
                    "path": "/path/to/test1.pdf",
                    "metadata": {"source": "test1.pdf"}
                }
            ],
            "total": 1,
            "page": 1,
            "page_size": 20,
            "pages": 1
        }
        engine._vector_store.list_documents = AsyncMock(return_value=mock_docs)
        
        result = await engine.list(return_stats=False)
        
        # 不应包含chunks_count和total_size
        assert "chunks_count" not in result["documents"][0]
        assert "total_size" not in result["documents"][0]

    @pytest.mark.asyncio
    async def test_list_with_name_filter(self, engine):
        """测试按文件名过滤"""
        all_docs = [
            {"name": "report_2024.pdf", "path": "/path/to/report_2024.pdf"},
            {"name": "summary_2024.pdf", "path": "/path/to/summary_2024.pdf"},
            {"name": "notes.md", "path": "/path/to/notes.md"}
        ]
        
        async def mock_filter_by_name(filter=None, **kwargs):
            if filter and "name_pattern" in filter:
                pattern = filter["name_pattern"]
                filtered = [doc for doc in all_docs if pattern in doc["name"]]
                return {
                    "documents": filtered,
                    "total": len(filtered),
                    "page": 1,
                    "page_size": 20,
                    "pages": 1
                }
            return {
                "documents": all_docs,
                "total": len(all_docs),
                "page": 1,
                "page_size": 20,
                "pages": 1
            }
        
        engine._vector_store.list_documents = mock_filter_by_name
        
        # 搜索包含"report"的文档
        result = await engine.list(filter={"name_pattern": "report"})
        assert len(result["documents"]) == 1
        assert "report" in result["documents"][0]["name"]

    @pytest.mark.asyncio
    async def test_list_error_handling(self, engine):
        """测试错误处理"""
        engine._vector_store.list_documents = AsyncMock(
            side_effect=Exception("Database error")
        )
        
        with pytest.raises(Exception, match="Database error"):
            await engine.list()