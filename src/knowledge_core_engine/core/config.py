"""Configuration for RAG system."""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """RAG system configuration.
    
    Example:
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            vectordb_provider="chromadb"
        )
    """
    
    # Provider selection
    llm_provider: str = "deepseek"
    embedding_provider: str = "dashscope"  
    vectordb_provider: str = "chromadb"
    
    # API keys (auto-loaded from env if not provided)
    llm_api_key: Optional[str] = None
    embedding_api_key: Optional[str] = None
    vectordb_api_key: Optional[str] = None
    
    # Model names (auto-set based on provider if not provided)
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    
    # LLM parameters
    temperature: float = 0.1
    max_tokens: int = 2048
    
    # Embedding parameters
    embedding_dimensions: Optional[int] = None
    embedding_batch_size: int = 25
    
    # VectorDB parameters
    collection_name: str = "knowledge_core"
    persist_directory: str = "./data/chroma_db"
    
    # Retrieval parameters
    retrieval_strategy: str = "hybrid"  # vector, bm25, hybrid
    retrieval_top_k: int = 10
    reranker_model: Optional[str] = None
    reranker_provider: Optional[str] = None
    reranker_api_key: Optional[str] = None
    
    # Features
    use_multi_vector: bool = True
    include_citations: bool = True
    
    # Advanced options
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Auto-configure defaults based on providers."""
        # Load API keys from environment
        if not self.llm_api_key:
            self.llm_api_key = os.getenv(f"{self.llm_provider.upper()}_API_KEY")
        
        if not self.embedding_api_key:
            if self.embedding_provider == "dashscope":
                self.embedding_api_key = os.getenv("DASHSCOPE_API_KEY")
            else:
                self.embedding_api_key = os.getenv(f"{self.embedding_provider.upper()}_API_KEY")
        
        if not self.vectordb_api_key and self.vectordb_provider != "chromadb":
            self.vectordb_api_key = os.getenv(f"{self.vectordb_provider.upper()}_API_KEY")
        
        # Set default models
        if not self.llm_model:
            self.llm_model = self._get_default_llm_model()
        
        if not self.embedding_model:
            self.embedding_model = self._get_default_embedding_model()
        
        if not self.embedding_dimensions:
            self.embedding_dimensions = self._get_default_dimensions()
        
        # Set default reranker if not specified
        if not self.reranker_model and self.retrieval_strategy in ["hybrid"]:
            self.reranker_model = "bge-reranker-v2-m3-qwen"
            self.reranker_provider = "huggingface"
        
        # Load reranker API key if needed
        if self.reranker_provider and not self.reranker_api_key:
            self.reranker_api_key = os.getenv(f"{self.reranker_provider.upper()}_API_KEY")
    
    def _get_default_llm_model(self) -> str:
        """Get default model for LLM provider."""
        defaults = {
            "deepseek": "deepseek-chat",
            "qwen": "qwen2.5-72b-instruct",
            "openai": "gpt-4-turbo-preview",
            "claude": "claude-3-opus-20240229"
        }
        return defaults.get(self.llm_provider, "unknown")
    
    def _get_default_embedding_model(self) -> str:
        """Get default model for embedding provider."""
        defaults = {
            "dashscope": "text-embedding-v3",
            "openai": "text-embedding-3-large",
            "cohere": "embed-multilingual-v3.0",
            "huggingface": "BAAI/bge-large-zh-v1.5"
        }
        return defaults.get(self.embedding_provider, "unknown")
    
    def _get_default_dimensions(self) -> int:
        """Get default dimensions for embedding model."""
        defaults = {
            "dashscope": 1536,
            "openai": 3072,
            "cohere": 1024,
            "huggingface": 1024
        }
        return defaults.get(self.embedding_provider, 1536)
    
    def validate(self) -> None:
        """Validate configuration."""
        # Validate provider names first
        valid_llm = ["deepseek", "qwen", "openai", "claude", "local"]
        valid_embedding = ["dashscope", "openai", "cohere", "huggingface", "local"]
        valid_vectordb = ["chromadb", "pinecone", "weaviate", "qdrant"]
        
        if self.llm_provider not in valid_llm:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}. Valid: {valid_llm}")
        
        if self.embedding_provider not in valid_embedding:
            raise ValueError(f"Invalid embedding provider: {self.embedding_provider}. Valid: {valid_embedding}")
        
        if self.vectordb_provider not in valid_vectordb:
            raise ValueError(f"Invalid vector DB provider: {self.vectordb_provider}. Valid: {valid_vectordb}")
        
        # Then check required API keys
        if self.llm_provider != "local" and not self.llm_api_key:
            raise ValueError(f"API key required for {self.llm_provider}")
        
        if self.embedding_provider not in ["huggingface", "local"] and not self.embedding_api_key:
            raise ValueError(f"API key required for {self.embedding_provider}")