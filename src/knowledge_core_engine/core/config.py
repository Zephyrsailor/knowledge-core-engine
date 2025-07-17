"""Configuration for RAG system.

This module provides the main configuration class for RAG-specific settings.
It is the primary configuration interface for the KnowledgeCore Engine.

Usage:
    from knowledge_core_engine.core.config import RAGConfig
    
    config = RAGConfig(
        llm_provider="deepseek",
        embedding_provider="dashscope",
        enable_reranking=True
    )

For environment settings (API keys, paths), see utils.config.get_settings().
"""

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
    llm_provider: str = "qwen"
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
    
    # Chunking parameters
    enable_hierarchical_chunking: bool = False  # Enable parent-child chunk relationships
    enable_semantic_chunking: bool = True       # Enable semantic-aware chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    enable_metadata_enhancement: bool = False   # Enable LLM-based metadata generation
    
    # Retrieval parameters
    retrieval_strategy: str = "hybrid"  # vector, bm25, hybrid
    retrieval_top_k: int = 10
    
    # Hybrid retrieval parameters
    vector_weight: float = 0.7          # Weight for vector search (0-1)
    bm25_weight: float = 0.3           # Weight for BM25 search (0-1)
    fusion_method: str = "weighted"     # weighted, rrf (reciprocal rank fusion)
    
    # BM25 parameters
    bm25_provider: str = "bm25s"        # bm25s, elasticsearch, none
    bm25_k1: float = 1.5               # BM25 k1 parameter
    bm25_b: float = 0.75               # BM25 b parameter
    bm25_epsilon: float = 0.25         # BM25 epsilon parameter
    language: str = "en"               # Language for tokenization (en, zh, multi)
    
    # Elasticsearch parameters (if using elasticsearch provider)
    elasticsearch_url: Optional[str] = None
    elasticsearch_index: str = "knowledge_core"
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[str] = None
    
    # Query expansion parameters
    enable_query_expansion: bool = False
    query_expansion_method: str = "llm"  # llm, rule_based
    query_expansion_count: int = 3       # Number of expanded queries
    
    # Reranking parameters
    enable_reranking: bool = False
    reranker_provider: str = "huggingface"  # huggingface, api, none
    reranker_model: Optional[str] = None    # Model name (e.g., bge-reranker-v2-m3, qwen3-reranker-8b)
    reranker_api_provider: Optional[str] = None  # dashscope, cohere, jina (for api provider)
    reranker_api_key: Optional[str] = None
    rerank_top_k: int = 5               # Number of documents to keep after reranking
    use_fp16: bool = True               # Use half precision for HuggingFace models
    reranker_device: Optional[str] = None  # Device for HuggingFace models (None = auto)
    
    # Citation parameters
    include_citations: bool = True
    citation_style: str = "inline"       # inline, footnote, endnote
    
    # Multi-vector parameters
    use_multi_vector: bool = True        # Index multiple representations per chunk
    
    # Deprecated - will be removed in future version
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
        
        # Set default reranker if reranking is enabled
        if self.enable_reranking and not self.reranker_model:
            self.reranker_model = "bge-reranker-v2-m3"
        
        # Set reranker API provider based on model if not specified
        if self.reranker_provider == "api" and not self.reranker_api_provider:
            if self.reranker_model and "gte-rerank" in self.reranker_model:
                self.reranker_api_provider = "dashscope"
            else:
                self.reranker_api_provider = "dashscope"  # Default
        
        # Load reranker API key if needed
        if self.reranker_provider == "api" and self.reranker_api_provider:
            if not self.reranker_api_key:
                self.reranker_api_key = os.getenv(f"{self.reranker_api_provider.upper()}_API_KEY")
        
        # Set default retrieval strategy based on BM25 provider
        if self.bm25_provider == "none" and self.retrieval_strategy == "hybrid":
            self.retrieval_strategy = "vector"  # Fallback to vector-only
        
        # Load Elasticsearch credentials if needed
        if self.bm25_provider == "elasticsearch":
            if not self.elasticsearch_username:
                self.elasticsearch_username = os.getenv("ELASTICSEARCH_USERNAME")
            if not self.elasticsearch_password:
                self.elasticsearch_password = os.getenv("ELASTICSEARCH_PASSWORD")
    
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
        
        # Validate new parameters
        if self.retrieval_strategy not in ["vector", "bm25", "hybrid"]:
            raise ValueError(f"Invalid retrieval strategy: {self.retrieval_strategy}")
        
        if self.fusion_method not in ["weighted", "rrf"]:
            raise ValueError(f"Invalid fusion method: {self.fusion_method}")
        
        if self.query_expansion_method not in ["llm", "rule_based"]:
            raise ValueError(f"Invalid query expansion method: {self.query_expansion_method}")
        
        if self.citation_style not in ["inline", "footnote", "endnote"]:
            raise ValueError(f"Invalid citation style: {self.citation_style}")
        
        # Validate BM25 parameters
        if self.bm25_provider not in ["bm25s", "elasticsearch", "none"]:
            raise ValueError(f"Invalid BM25 provider: {self.bm25_provider}")
        
        if self.bm25_provider == "elasticsearch" and not self.elasticsearch_url:
            raise ValueError("elasticsearch_url required when using elasticsearch provider")
        
        # Validate reranker parameters
        if self.reranker_provider not in ["huggingface", "api", "none"]:
            raise ValueError(f"Invalid reranker provider: {self.reranker_provider}")
        
        if self.reranker_provider == "api" and not self.reranker_api_provider:
            raise ValueError("reranker_api_provider required when using api provider")
        
        if self.reranker_api_provider and self.reranker_api_provider not in ["dashscope", "cohere", "jina"]:
            raise ValueError(f"Invalid reranker API provider: {self.reranker_api_provider}")
        
        # Validate weights
        if not 0 <= self.vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        
        if not 0 <= self.bm25_weight <= 1:
            raise ValueError("bm25_weight must be between 0 and 1")
        
        if self.retrieval_strategy == "hybrid" and abs(self.vector_weight + self.bm25_weight - 1.0) > 0.01:
            raise ValueError("For hybrid retrieval, vector_weight + bm25_weight should equal 1.0")
        
        # Validate chunk parameters
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")