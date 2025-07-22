"""Tests for enhanced RAG configuration."""

import pytest
import os
from knowledge_core_engine.core.config import RAGConfig


class TestEnhancedRAGConfig:
    """Test enhanced configuration options."""
    
    def test_chunking_configuration(self):
        """Test chunking-related configuration."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            enable_hierarchical_chunking=True,
            enable_semantic_chunking=False,
            chunk_size=1024,
            chunk_overlap=100,
            enable_metadata_enhancement=True
        )
        
        assert config.enable_hierarchical_chunking is True
        assert config.enable_semantic_chunking is False
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.enable_metadata_enhancement is True
    
    def test_retrieval_strategy_configuration(self):
        """Test retrieval strategy configuration."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            retrieval_strategy="hybrid",
            vector_weight=0.6,
            bm25_weight=0.4,
            fusion_method="rrf"
        )
        
        assert config.retrieval_strategy == "hybrid"
        assert config.vector_weight == 0.6
        assert config.bm25_weight == 0.4
        assert config.fusion_method == "rrf"
    
    def test_query_expansion_configuration(self):
        """Test query expansion configuration."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            enable_query_expansion=True,
            query_expansion_method="llm",
            query_expansion_count=5
        )
        
        assert config.enable_query_expansion is True
        assert config.query_expansion_method == "llm"
        assert config.query_expansion_count == 5
    
    def test_reranking_configuration(self):
        """Test reranking configuration."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            enable_reranking=True,
            reranker_model="bge-reranker-large",
            reranker_provider="huggingface",
            rerank_top_k=3
        )
        
        assert config.enable_reranking is True
        assert config.reranker_model == "bge-reranker-large"
        assert config.reranker_provider == "huggingface"
        assert config.rerank_top_k == 3
    
    def test_default_reranker_when_enabled(self):
        """Test default reranker is set when reranking is enabled."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            enable_reranking=True
        )
        
        # Should set default reranker
        assert config.reranker_model == "bge-reranker-v2-m3"
        assert config.reranker_provider == "huggingface"
    
    def test_validation_retrieval_strategy(self):
        """Test validation of retrieval strategy."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            retrieval_strategy="invalid"
        )
        with pytest.raises(ValueError, match="Invalid retrieval strategy"):
            config.validate()
    
    def test_validation_fusion_method(self):
        """Test validation of fusion method."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            fusion_method="invalid"
        )
        with pytest.raises(ValueError, match="Invalid fusion method"):
            config.validate()
    
    def test_validation_query_expansion_method(self):
        """Test validation of query expansion method."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            query_expansion_method="invalid"
        )
        with pytest.raises(ValueError, match="Invalid query expansion method"):
            config.validate()
    
    def test_validation_weights(self):
        """Test validation of weight parameters."""
        # Vector weight out of range
        config1 = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            vector_weight=1.5
        )
        with pytest.raises(ValueError, match="vector_weight must be between 0 and 1"):
            config1.validate()
        
        # BM25 weight out of range
        config2 = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            bm25_weight=-0.1
        )
        with pytest.raises(ValueError, match="bm25_weight must be between 0 and 1"):
            config2.validate()
    
    def test_validation_hybrid_weights_sum(self):
        """Test validation of hybrid retrieval weights sum."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            retrieval_strategy="hybrid",
            vector_weight=0.3,
            bm25_weight=0.3
        )
        with pytest.raises(ValueError, match="vector_weight \\+ bm25_weight should equal 1.0"):
            config.validate()
    
    def test_validation_chunk_parameters(self):
        """Test validation of chunk parameters."""
        # Negative chunk size
        config1 = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            chunk_size=-100
        )
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config1.validate()
        
        # Negative overlap
        config2 = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            chunk_overlap=-50
        )
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            config2.validate()
        
        # Overlap >= chunk size
        config3 = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            chunk_size=100,
            chunk_overlap=150
        )
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            config3.validate()
    
    def test_backward_compatibility(self):
        """Test that old configuration still works."""
        # Should not raise any errors
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            temperature=0.2,
            max_tokens=1024,
            retrieval_top_k=5,
            include_citations=True
        )
        
        # Check defaults are set
        assert config.enable_hierarchical_chunking is False
        assert config.enable_semantic_chunking is True
        assert config.retrieval_strategy == "hybrid"
        assert config.enable_query_expansion is False
        assert config.enable_reranking is False
    
    def test_environment_variable_loading(self):
        """Test loading API keys from environment."""
        # Set environment variable with KCE_ prefix
        os.environ["KCE_DEEPSEEK_API_KEY"] = "test-deepseek-key"
        os.environ["KCE_DASHSCOPE_API_KEY"] = "test-dashscope-key"
        
        try:
            config = RAGConfig(
                llm_provider="deepseek",
                embedding_provider="dashscope"
            )
            
            # Check that API keys were loaded from environment
            assert config.llm_api_key == "test-deepseek-key"
            assert config.embedding_api_key == "test-dashscope-key"
        finally:
            # Clean up
            if "KCE_DEEPSEEK_API_KEY" in os.environ:
                del os.environ["KCE_DEEPSEEK_API_KEY"]
            if "KCE_DASHSCOPE_API_KEY" in os.environ:
                del os.environ["KCE_DASHSCOPE_API_KEY"]
    
    def test_comprehensive_configuration(self):
        """Test a comprehensive configuration with all features enabled."""
        config = RAGConfig(
            # Providers
            llm_provider="deepseek",
            embedding_provider="dashscope",
            vectordb_provider="chromadb",
            
            # Chunking
            enable_hierarchical_chunking=True,
            enable_semantic_chunking=True,
            chunk_size=1024,
            chunk_overlap=128,
            enable_metadata_enhancement=True,
            
            # Retrieval
            retrieval_strategy="hybrid",
            retrieval_top_k=20,
            vector_weight=0.7,
            bm25_weight=0.3,
            fusion_method="weighted",
            
            # Query expansion
            enable_query_expansion=True,
            query_expansion_method="llm",
            query_expansion_count=3,
            
            # Reranking
            enable_reranking=True,
            reranker_model="bge-reranker-v2-m3",
            reranker_provider="huggingface",
            rerank_top_k=5,
            
            # Citations
            include_citations=True,
            citation_style="inline",
            
            # Multi-vector
            use_multi_vector=True
        )
        
        # Validate all settings are properly configured
        config.validate()
        
        # Check all values
        assert config.enable_hierarchical_chunking is True
        assert config.enable_metadata_enhancement is True
        assert config.retrieval_strategy == "hybrid"
        assert config.enable_query_expansion is True
        assert config.enable_reranking is True
        assert config.include_citations is True
    
    def test_deprecated_extra_params_warning(self):
        """Test that extra_params is deprecated but still works."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            extra_params={"custom_key": "custom_value"}
        )
        
        # Should still be accessible
        assert config.extra_params["custom_key"] == "custom_value"
        
        # But new parameters should be used instead
        assert hasattr(config, 'enable_hierarchical_chunking')
        assert hasattr(config, 'enable_query_expansion')