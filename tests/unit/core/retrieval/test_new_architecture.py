"""Test the new provider architecture for BM25 and Reranker."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from knowledge_core_engine.core.config import RAGConfig

# BM25 imports
from knowledge_core_engine.core.retrieval.bm25.base import BaseBM25Retriever, BM25Result
from knowledge_core_engine.core.retrieval.bm25.bm25s_retriever import BM25SRetriever
from knowledge_core_engine.core.retrieval.bm25.factory import create_bm25_retriever

# Reranker imports
from knowledge_core_engine.core.retrieval.reranker.base import BaseReranker, RerankResult
from knowledge_core_engine.core.retrieval.reranker.factory import create_reranker


class TestBM25Factory:
    """Test BM25 factory and providers."""
    
    def test_create_bm25s_retriever(self):
        """Test creating BM25S retriever."""
        config = RAGConfig(
            retrieval_strategy="hybrid",
            bm25_provider="bm25s",
            language="en"
        )
        
        retriever = create_bm25_retriever(config)
        
        assert retriever is not None
        assert isinstance(retriever, BM25SRetriever)
        assert retriever.language == "en"
    
    def test_create_bm25_none_for_vector_only(self):
        """Test that BM25 is not created for vector-only strategy."""
        config = RAGConfig(
            retrieval_strategy="vector",
            bm25_provider="bm25s"
        )
        
        retriever = create_bm25_retriever(config)
        
        assert retriever is None
    
    def test_create_elasticsearch_requires_url(self):
        """Test that Elasticsearch requires URL."""
        config = RAGConfig(
            retrieval_strategy="hybrid",
            bm25_provider="elasticsearch"
            # No elasticsearch_url provided
        )
        
        with pytest.raises(ValueError, match="Elasticsearch URL not provided"):
            create_bm25_retriever(config)


class TestRerankerFactory:
    """Test Reranker factory and providers."""
    
    def test_create_reranker_disabled(self):
        """Test that reranker is not created when disabled."""
        config = RAGConfig(
            enable_reranking=False
        )
        
        reranker = create_reranker(config)
        
        assert reranker is None
    
    def test_create_huggingface_reranker(self):
        """Test creating HuggingFace reranker."""
        config = RAGConfig(
            enable_reranking=True,
            reranker_provider="huggingface",
            reranker_model="bge-reranker-v2-m3"
        )
        
        with patch('knowledge_core_engine.core.retrieval.reranker.factory.HuggingFaceReranker') as mock_hf:
            reranker = create_reranker(config)
            
            mock_hf.assert_called_once_with(
                model_name="bge-reranker-v2-m3",
                use_fp16=True,
                device=None
            )
    
    def test_create_api_reranker_dashscope(self):
        """Test creating API reranker for DashScope."""
        config = RAGConfig(
            enable_reranking=True,
            reranker_provider="api",
            reranker_model="gte-rerank-v2"
        )
        
        with patch('knowledge_core_engine.core.retrieval.reranker.factory.APIReranker') as mock_api:
            reranker = create_reranker(config)
            
            mock_api.assert_called_once_with(
                provider="dashscope",
                api_key=None,
                model="gte-rerank-v2",
                timeout=30
            )


@pytest.mark.asyncio
class TestBM25SRetriever:
    """Test BM25S retriever implementation."""
    
    async def test_bm25s_basic_functionality(self):
        """Test basic BM25S functionality."""
        retriever = BM25SRetriever(language="en")
        
        # Mock bm25s module
        with patch.object(retriever, '_bm25s') as mock_bm25s:
            # Set up mocks
            mock_bm25s.tokenize.return_value = [["test", "document"]]
            mock_bm25s.BM25.return_value.retrieve.return_value = ([0], [0.5])
            
            await retriever.initialize()
            
            # Add documents
            await retriever.add_documents(
                ["test document"],
                ["doc1"],
                [{"source": "test"}]
            )
            
            # Search
            results = await retriever.search("test", top_k=1)
            
            assert len(results) == 1
            assert results[0].document_id == "doc1"
            assert results[0].score == 0.5


@pytest.mark.asyncio
class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = RAGConfig(
            retrieval_strategy="hybrid",
            bm25_provider="bm25s",
            reranker_provider="huggingface",
            reranker_model="bge-reranker-v2-m3"
        )
        
        # Should not raise
        config.validate()
    
    def test_invalid_bm25_provider(self):
        """Test invalid BM25 provider."""
        config = RAGConfig(
            bm25_provider="invalid"
        )
        
        with pytest.raises(ValueError, match="Invalid BM25 provider"):
            config.validate()
    
    def test_invalid_reranker_provider(self):
        """Test invalid reranker provider."""
        config = RAGConfig(
            reranker_provider="invalid"
        )
        
        with pytest.raises(ValueError, match="Invalid reranker provider"):
            config.validate()
    
    def test_elasticsearch_requires_url(self):
        """Test Elasticsearch requires URL."""
        config = RAGConfig(
            bm25_provider="elasticsearch"
        )
        
        with pytest.raises(ValueError, match="elasticsearch_url required"):
            config.validate()


class TestIntegration:
    """Test integration of new architecture."""
    
    @pytest.mark.asyncio
    async def test_retriever_with_new_bm25(self):
        """Test that retriever works with new BM25 system."""
        from knowledge_core_engine.core.retrieval.bm25_retriever import BM25Retriever
        
        config = RAGConfig(
            retrieval_strategy="bm25",
            bm25_provider="bm25s"
        )
        
        retriever = BM25Retriever(config)
        
        # Should be able to initialize
        await retriever.initialize()
        
        assert retriever._retriever is not None
    
    @pytest.mark.asyncio
    async def test_reranker_with_new_system(self):
        """Test that reranker works with new system."""
        from knowledge_core_engine.core.retrieval.reranker import Reranker
        
        config = RAGConfig(
            enable_reranking=True,
            reranker_provider="huggingface",
            reranker_model="bge-reranker-v2-m3"
        )
        
        reranker = Reranker(config)
        
        # Mock the factory
        with patch('knowledge_core_engine.core.retrieval.reranker.create_reranker') as mock_factory:
            mock_reranker = AsyncMock(spec=BaseReranker)
            mock_factory.return_value = mock_reranker
            
            await reranker.initialize()
            
            assert reranker._reranker is not None