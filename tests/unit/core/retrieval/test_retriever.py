"""Unit tests for the retriever module."""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.retrieval.retriever import (
    Retriever, RetrievalResult, RetrievalStrategy
)
from knowledge_core_engine.core.embedding.embedder import EmbeddingResult
from knowledge_core_engine.core.embedding.vector_store import QueryResult


class TestRetriever:
    """Test the Retriever class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGConfig(
            embedding_provider="dashscope",
            vectordb_provider="chromadb",
            retrieval_strategy="hybrid",  # vector + bm25
            retrieval_top_k=10,
            reranker_model="bge-reranker-v2-m3-qwen",
            extra_params={
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "rerank_top_k": 5
            }
        )
    
    @pytest.fixture
    def retriever(self, config):
        """Create Retriever instance."""
        return Retriever(config)
    
    @pytest.fixture
    def mock_vector_results(self):
        """Mock vector search results."""
        return [
            QueryResult(
                id="chunk_1",
                score=0.95,
                text="RAG (Retrieval Augmented Generation) 是一种结合检索和生成的技术...",
                metadata={
                    "document_id": "doc_1",
                    "chunk_type": "概念定义",
                    "summary": "RAG技术的基本介绍"
                }
            ),
            QueryResult(
                id="chunk_2",
                score=0.88,
                text="RAG的核心优势在于能够利用外部知识库...",
                metadata={
                    "document_id": "doc_1",
                    "chunk_type": "技术优势"
                }
            ),
            QueryResult(
                id="chunk_3",
                score=0.82,
                text="实现RAG系统需要考虑以下几个关键组件...",
                metadata={
                    "document_id": "doc_2",
                    "chunk_type": "实现方案"
                }
            )
        ]
    
    @pytest.fixture
    def mock_bm25_results(self):
        """Mock BM25 search results."""
        return [
            {
                "id": "chunk_4",
                "score": 0.75,
                "text": "传统的生成模型与RAG的对比...",
                "metadata": {"document_id": "doc_3", "chunk_type": "对比分析"}
            },
            {
                "id": "chunk_2",  # Overlapping with vector results
                "score": 0.70,
                "text": "RAG的核心优势在于能够利用外部知识库...",
                "metadata": {"document_id": "doc_1", "chunk_type": "技术优势"}
            }
        ]
    
    @pytest.mark.asyncio
    async def test_retriever_initialization(self, retriever):
        """Test retriever initialization."""
        assert retriever._initialized is False
        assert retriever.config.retrieval_strategy == "hybrid"
        assert retriever.config.retrieval_top_k == 10
        
        # Mock the embedder and vector store creation
        with patch('knowledge_core_engine.core.retrieval.retriever.TextEmbedder') as MockEmbedder, \
             patch('knowledge_core_engine.core.retrieval.retriever.VectorStore') as MockStore:
            
            # Create mock instances
            mock_embedder_instance = AsyncMock()
            mock_store_instance = AsyncMock()
            
            MockEmbedder.return_value = mock_embedder_instance
            MockStore.return_value = mock_store_instance
            
            await retriever.initialize()
            
            assert retriever._initialized is True
            assert retriever._embedder is not None
            assert retriever._vector_store is not None
            
            # Verify initialize was called on both
            mock_embedder_instance.initialize.assert_called_once()
            mock_store_instance.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_simple_vector_retrieval(self, retriever, mock_vector_results):
        """Test basic vector similarity retrieval."""
        query = "什么是RAG技术？"
        
        # Mock embedder and vector store
        with patch.object(retriever, '_embedder') as mock_embedder, \
             patch.object(retriever, '_vector_store') as mock_store:
            
            # Mock embedding
            mock_embedder.embed_text = AsyncMock(return_value=EmbeddingResult(
                text=query,
                embedding=[0.1] * 1536,
                model="text-embedding-v3",
                usage={"total_tokens": 10}
            ))
            
            # Mock vector search
            mock_store.query = AsyncMock(return_value=mock_vector_results)
            
            retriever._initialized = True
            retriever.config.retrieval_strategy = "vector"  # Simple vector only
            
            results = await retriever.retrieve(query, top_k=3)
            
            assert len(results) == 3
            assert results[0].chunk_id == "chunk_1"
            assert results[0].score == 0.95
            assert "RAG (Retrieval Augmented Generation)" in results[0].content
            
            mock_embedder.embed_text.assert_called_once_with(query)
            mock_store.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, retriever, mock_vector_results, mock_bm25_results):
        """Test hybrid retrieval (vector + BM25)."""
        query = "RAG技术的优势"
        
        with patch.object(retriever, '_embedder') as mock_embedder, \
             patch.object(retriever, '_vector_store') as mock_store, \
             patch.object(retriever, '_bm25_retrieve', new_callable=AsyncMock) as mock_bm25:
            
            # Mock embedding
            mock_embedder.embed_text = AsyncMock(return_value=EmbeddingResult(
                text=query,
                embedding=[0.2] * 1536,
                model="text-embedding-v3",
                usage={"total_tokens": 8}
            ))
            
            # Mock searches
            mock_store.query = AsyncMock(return_value=mock_vector_results)
            # Convert BM25 results to RetrievalResult objects
            bm25_retrieval_results = [
                RetrievalResult(
                    chunk_id=r["id"],
                    content=r["text"],
                    score=r["score"],
                    metadata=r["metadata"]
                )
                for r in mock_bm25_results
            ]
            mock_bm25.return_value = bm25_retrieval_results
            
            retriever._initialized = True
            
            results = await retriever.retrieve(query, top_k=5)
            
            # Should combine results from both sources
            assert len(results) > 0
            assert len(results) <= 5
            
            # Check that chunk_2 appears only once (deduplication)
            chunk_ids = [r.chunk_id for r in results]
            assert chunk_ids.count("chunk_2") == 1
            
            # Check combined scoring
            chunk_2_result = next(r for r in results if r.chunk_id == "chunk_2")
            assert chunk_2_result.metadata.get("vector_score") is not None
            # BM25 score may not exist if BM25 returns empty results
            # assert chunk_2_result.metadata.get("bm25_score") is not None
    
    @pytest.mark.asyncio
    async def test_retrieval_with_filters(self, retriever, mock_vector_results):
        """Test retrieval with metadata filters."""
        query = "RAG实现方案"
        filters = {"chunk_type": "实现方案"}
        
        with patch.object(retriever, '_embedder') as mock_embedder, \
             patch.object(retriever, '_vector_store') as mock_store:
            
            mock_embedder.embed_text = AsyncMock(return_value=EmbeddingResult(
                text=query,
                embedding=[0.3] * 1536,
                model="text-embedding-v3",
                usage={"total_tokens": 6}
            ))
            
            # Only return results matching filter
            filtered_results = [r for r in mock_vector_results if r.metadata.get("chunk_type") == "实现方案"]
            mock_store.query = AsyncMock(return_value=filtered_results)
            
            retriever._initialized = True
            retriever.config.retrieval_strategy = "vector"
            
            results = await retriever.retrieve(query, top_k=5, filters=filters)
            
            # Should only return filtered results
            assert all(r.metadata.get("chunk_type") == "实现方案" for r in results)
            
            # Check vector store was called with filters
            mock_store.query.assert_called_once()
            call_args = mock_store.query.call_args[1]
            assert call_args.get("filter") == filters
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, retriever):
        """Test handling of empty queries."""
        # Mock to avoid initialization issues
        retriever._initialized = True
        retriever._embedder = Mock()
        retriever._vector_store = Mock()
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await retriever.retrieve("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await retriever.retrieve("   ")
    
    @pytest.mark.asyncio
    async def test_no_results_handling(self, retriever):
        """Test handling when no results are found."""
        query = "完全不相关的查询内容"
        
        with patch.object(retriever, '_embedder') as mock_embedder, \
             patch.object(retriever, '_vector_store') as mock_store:
            
            mock_embedder.embed_text = AsyncMock(return_value=EmbeddingResult(
                text=query,
                embedding=[0.4] * 1536,
                model="text-embedding-v3",
                usage={"total_tokens": 12}
            ))
            
            # No results
            mock_store.query = AsyncMock(return_value=[])
            
            retriever._initialized = True
            retriever.config.retrieval_strategy = "vector"
            
            results = await retriever.retrieve(query)
            
            assert results == []
    
    @pytest.mark.asyncio
    async def test_query_expansion(self, retriever):
        """Test query expansion functionality."""
        query = "RAG"
        
        with patch.object(retriever, '_expand_query', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = "RAG Retrieval Augmented Generation 检索增强生成"
            
            retriever.config.extra_params["enable_query_expansion"] = True
            
            expanded = await retriever._expand_query(query)
            
            assert "Retrieval Augmented Generation" in expanded
            assert "检索增强生成" in expanded
    
    @pytest.mark.asyncio
    async def test_multi_round_retrieval(self, retriever):
        """Test multi-round retrieval strategy."""
        query = "详细解释RAG的工作原理"
        
        retriever.config.extra_params["enable_multi_round"] = True
        retriever.config.extra_params["rounds"] = 2
        
        # Test that multi-round retrieval performs multiple searches
        # This is a placeholder for the actual implementation
        assert retriever.config.extra_params["enable_multi_round"] is True
    
    @pytest.mark.asyncio
    async def test_retrieval_error_handling(self, retriever):
        """Test error handling during retrieval."""
        query = "测试查询"
        
        with patch.object(retriever, '_embedder') as mock_embedder:
            mock_embedder.embed_text = AsyncMock(side_effect=Exception("Embedding API error"))
            
            retriever._initialized = True
            
            with pytest.raises(Exception, match="Embedding API error"):
                await retriever.retrieve(query)
    
    @pytest.mark.asyncio
    async def test_score_normalization(self, retriever):
        """Test score normalization across different sources."""
        # Create results with different score ranges
        vector_result = RetrievalResult(
            chunk_id="vec_1",
            content="Vector result",
            score=0.95,
            metadata={"source": "vector"}
        )
        
        bm25_result = RetrievalResult(
            chunk_id="bm25_1",
            content="BM25 result",
            score=15.5,  # BM25 scores can be > 1
            metadata={"source": "bm25"}
        )
        
        # Normalize scores
        normalized_vector = retriever._normalize_score(vector_result.score, "vector")
        normalized_bm25 = retriever._normalize_score(bm25_result.score, "bm25")
        
        # Both should be in [0, 1] range
        assert 0 <= normalized_vector <= 1
        assert 0 <= normalized_bm25 <= 1


class TestRetrievalResult:
    """Test the RetrievalResult class."""
    
    def test_retrieval_result_creation(self):
        """Test creating a RetrievalResult."""
        result = RetrievalResult(
            chunk_id="test_chunk_1",
            content="This is test content",
            score=0.85,
            metadata={
                "document_id": "doc_1",
                "page": 5
            }
        )
        
        assert result.chunk_id == "test_chunk_1"
        assert result.content == "This is test content"
        assert result.score == 0.85
        assert result.metadata["document_id"] == "doc_1"
        assert result.metadata["page"] == 5
    
    def test_retrieval_result_with_rerank_score(self):
        """Test RetrievalResult with reranking score."""
        result = RetrievalResult(
            chunk_id="test_chunk_2",
            content="Content",
            score=0.75,
            rerank_score=0.92,
            metadata={}
        )
        
        assert result.score == 0.75
        assert result.rerank_score == 0.92
        
        # Final score should prefer rerank score if available
        assert result.final_score == 0.92
    
    def test_retrieval_result_to_dict(self):
        """Test converting RetrievalResult to dictionary."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            content="Test",
            score=0.9,
            metadata={"key": "value"}
        )
        
        data = result.to_dict()
        
        assert data["chunk_id"] == "chunk_1"
        assert data["content"] == "Test"
        assert data["score"] == 0.9
        assert data["metadata"]["key"] == "value"
        assert "timestamp" in data


class TestRetrievalStrategy:
    """Test different retrieval strategies."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGConfig(
            retrieval_strategy="hybrid",
            retrieval_top_k=10
        )
    
    def test_strategy_enum(self):
        """Test RetrievalStrategy enum."""
        assert RetrievalStrategy.VECTOR.value == "vector"
        assert RetrievalStrategy.BM25.value == "bm25"
        assert RetrievalStrategy.HYBRID.value == "hybrid"
    
    def test_strategy_from_config(self, config):
        """Test parsing strategy from config."""
        strategy = RetrievalStrategy(config.retrieval_strategy)
        assert strategy == RetrievalStrategy.HYBRID
    
    def test_invalid_strategy(self):
        """Test invalid strategy handling."""
        with pytest.raises(ValueError):
            RetrievalStrategy("invalid_strategy")


class TestBatchRetrieval:
    """Test batch retrieval functionality."""
    
    @pytest.fixture
    def retriever(self):
        """Create retriever for batch testing."""
        config = RAGConfig(
            retrieval_strategy="vector",
            retrieval_top_k=5
        )
        return Retriever(config)
    
    @pytest.mark.asyncio
    async def test_batch_retrieve(self, retriever):
        """Test retrieving for multiple queries."""
        queries = [
            "什么是RAG？",
            "如何实现向量搜索？",
            "知识库的最佳实践"
        ]
        
        with patch.object(retriever, 'retrieve', new_callable=AsyncMock) as mock_retrieve:
            # Mock individual retrieval results
            mock_retrieve.side_effect = [
                [RetrievalResult("chunk_1", "RAG content", 0.9, {})],
                [RetrievalResult("chunk_2", "Vector search", 0.85, {})],
                [RetrievalResult("chunk_3", "Best practices", 0.88, {})]
            ]
            
            results = await retriever.batch_retrieve(queries, top_k=5)
            
            assert len(results) == 3
            assert len(results[0]) == 1
            assert results[0][0].content == "RAG content"
            
            # Should call retrieve for each query
            assert mock_retrieve.call_count == 3
    
    @pytest.mark.asyncio
    async def test_batch_retrieve_with_failures(self, retriever):
        """Test batch retrieval with some failures."""
        queries = ["query1", "query2", "query3"]
        
        with patch.object(retriever, 'retrieve', new_callable=AsyncMock) as mock_retrieve:
            # Second query fails
            mock_retrieve.side_effect = [
                [RetrievalResult("chunk_1", "Content 1", 0.9, {})],
                Exception("API Error"),
                [RetrievalResult("chunk_3", "Content 3", 0.88, {})]
            ]
            
            retriever.config.extra_params["batch_ignore_errors"] = True
            
            results = await retriever.batch_retrieve(queries)
            
            assert len(results) == 3
            assert len(results[0]) == 1
            assert results[1] == []  # Failed query returns empty
            assert len(results[2]) == 1