"""Base classes and interfaces for embedding module.

This module defines the abstract interfaces that allow for flexible
implementation of different embedding providers and strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from enum import Enum


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    DASHSCOPE = "dashscope"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    CUSTOM = "custom"


class VectorStoreProvider(str, Enum):
    """Supported vector store providers."""
    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    CUSTOM = "custom"


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    text: str
    embedding: Union[List[float], np.ndarray]
    model: str = ""
    usage: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def vector_id(self) -> str:
        """Get unique ID for this vector."""
        return self.metadata.get("chunk_id", f"emb_{hash(self.text)}")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.embedding)


class IEmbedder(ABC):
    """Interface for text embedding providers."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of EmbeddingResults
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        pass


class IVectorStore(ABC):
    """Interface for vector storage systems."""
    
    @abstractmethod
    def add(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents with embeddings and metadata
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector store.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of results with scores and metadata
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    def update(self, documents: List[Dict[str, Any]]) -> None:
        """Update existing documents."""
        pass


class IEmbeddingStrategy(ABC):
    """Interface for embedding strategies."""
    
    @abstractmethod
    def prepare_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare text for embedding based on strategy.
        
        Args:
            content: Original text content
            metadata: Associated metadata (summary, questions, etc.)
            
        Returns:
            Prepared text for embedding
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass


@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    
    # API settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 30
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Performance settings
    batch_size: int = 25
    max_concurrent_requests: int = 10
    
    # Cache settings
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")