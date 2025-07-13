"""Tests for RAGConfig."""

import pytest
import os
from unittest.mock import patch

from knowledge_core_engine.core.config import RAGConfig


class TestRAGConfig:
    """Test RAGConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RAGConfig()
        
        # Provider defaults
        assert config.llm_provider == "deepseek"
        assert config.embedding_provider == "dashscope"
        assert config.vectordb_provider == "chromadb"
        
        # Model defaults
        assert config.llm_model == "deepseek-chat"
        assert config.embedding_model == "text-embedding-v3"
        assert config.embedding_dimensions == 1536
        
        # Parameters
        assert config.temperature == 0.1
        assert config.max_tokens == 2048
        assert config.embedding_batch_size == 25
        
        # Features
        assert config.use_multi_vector is True
        assert config.include_citations is True
    
    def test_custom_providers(self):
        """Test configuration with custom providers."""
        config = RAGConfig(
            llm_provider="qwen",
            embedding_provider="openai",
            vectordb_provider="pinecone"
        )
        
        assert config.llm_provider == "qwen"
        assert config.embedding_provider == "openai"
        assert config.vectordb_provider == "pinecone"
        
        # Should set appropriate defaults
        assert config.llm_model == "qwen2.5-72b-instruct"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.embedding_dimensions == 3072
    
    def test_api_key_from_env(self):
        """Test loading API keys from environment."""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-deepseek-key",
            "DASHSCOPE_API_KEY": "test-dashscope-key",
            "PINECONE_API_KEY": "test-pinecone-key"
        }):
            config = RAGConfig()
            
            assert config.llm_api_key == "test-deepseek-key"
            assert config.embedding_api_key == "test-dashscope-key"
            
            # Pinecone key only loaded if using pinecone
            config2 = RAGConfig(vectordb_provider="pinecone")
            assert config2.vectordb_api_key == "test-pinecone-key"
    
    def test_explicit_api_keys(self):
        """Test explicitly provided API keys."""
        config = RAGConfig(
            llm_api_key="explicit-llm-key",
            embedding_api_key="explicit-embed-key"
        )
        
        assert config.llm_api_key == "explicit-llm-key"
        assert config.embedding_api_key == "explicit-embed-key"
    
    def test_validation_missing_api_key(self):
        """Test validation fails with missing API key."""
        config = RAGConfig(llm_provider="openai")
        # Don't set API key
        
        with pytest.raises(ValueError, match="API key required for openai"):
            config.validate()
    
    def test_validation_invalid_provider(self):
        """Test validation with invalid provider names."""
        config = RAGConfig(llm_provider="invalid_llm")
        
        with pytest.raises(ValueError, match="Invalid LLM provider"):
            config.validate()
    
    def test_huggingface_no_api_key_required(self):
        """Test HuggingFace embedding doesn't require API key."""
        # Set a dummy LLM API key or use local LLM
        config = RAGConfig(
            embedding_provider="huggingface",
            llm_provider="local"  # Local LLM doesn't need API key
        )
        
        # Should not raise error
        config.validate()
        
        assert config.embedding_model == "BAAI/bge-large-zh-v1.5"
        assert config.embedding_dimensions == 1024
    
    def test_extra_params(self):
        """Test extra parameters."""
        config = RAGConfig(
            extra_params={
                "custom_option": "value",
                "another_option": 123
            }
        )
        
        assert config.extra_params["custom_option"] == "value"
        assert config.extra_params["another_option"] == 123
    
    def test_common_configurations(self):
        """Test common configuration patterns."""
        # Chinese-focused stack
        config1 = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            vectordb_provider="chromadb"
        )
        assert config1.llm_model == "deepseek-chat"
        assert config1.embedding_model == "text-embedding-v3"
        
        # OpenAI stack
        config2 = RAGConfig(
            llm_provider="openai",
            embedding_provider="openai",
            vectordb_provider="pinecone"
        )
        assert config2.llm_model == "gpt-4-turbo-preview"
        assert config2.embedding_model == "text-embedding-3-large"
        
        # Local/hybrid stack
        config3 = RAGConfig(
            llm_provider="qwen",
            embedding_provider="huggingface",
            vectordb_provider="chromadb"
        )
        assert config3.llm_model == "qwen2.5-72b-instruct"
        assert config3.embedding_model == "BAAI/bge-large-zh-v1.5"