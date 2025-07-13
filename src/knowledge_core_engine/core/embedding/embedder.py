"""Text embedding implementation with provider abstraction."""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import logging

from ..config import RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    text: str
    embedding: Union[List[float], np.ndarray]
    model: str
    usage: Dict[str, int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def vector_id(self) -> str:
        """Generate unique ID for this vector."""
        if self.metadata and "chunk_id" in self.metadata:
            return self.metadata["chunk_id"]
        return f"emb_{hashlib.md5(self.text.encode()).hexdigest()[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        import datetime
        return {
            "text": self.text,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "model": self.model,
            "usage": self.usage,
            "metadata": self.metadata,
            "vector_id": self.vector_id,
            "timestamp": datetime.datetime.now().isoformat()
        }


class TextEmbedder:
    """Text embedder with support for multiple providers."""
    
    def __init__(self, config: RAGConfig):
        """Initialize embedder with configuration.
        
        Args:
            config: RAG configuration object
        """
        self.config = config
        self._cache = {} if config.extra_params.get("enable_cache", True) else None
        self._provider = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the embedding provider."""
        if self._initialized:
            return
        
        # Create provider based on config
        if self.config.embedding_provider == "dashscope":
            self._provider = DashScopeProvider(self.config)
        elif self.config.embedding_provider == "openai":
            self._provider = OpenAIProvider(self.config)
        elif self.config.embedding_provider == "huggingface":
            self._provider = HuggingFaceProvider(self.config)
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embedding_provider}")
        
        await self._provider.initialize()
        self._initialized = True
        
        logger.info(f"Initialized {self.config.embedding_provider} embedding provider")
    
    async def embed_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        """Embed a single text.
        
        Args:
            text: Text to embed
            metadata: Optional metadata to attach
            
        Returns:
            EmbeddingResult with embedding vector
        """
        await self.initialize()
        
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        # Truncate text if too long
        max_length = self.config.extra_params.get("truncate_length", 6000)
        truncated = False
        if len(text) > max_length:
            text = text[:max_length]
            truncated = True
        
        # Check cache
        cache_key = self._get_cache_key(text)
        if self._cache and cache_key in self._cache:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            cached = self._cache[cache_key]
            result = EmbeddingResult(**cached)
            if metadata:
                result.metadata.update(metadata)
            return result
        
        # Get embedding from provider
        result = await self._provider.embed(text)
        
        # Add metadata
        if metadata:
            result.metadata.update(metadata)
        
        # Add truncation flag if text was truncated
        if truncated:
            result.metadata["truncated"] = True
        
        # Cache result
        if self._cache is not None:
            cache_data = result.to_dict()
            # Remove fields that are not part of constructor
            cache_data.pop('vector_id', None)
            cache_data.pop('timestamp', None)
            self._cache[cache_key] = cache_data
        
        return result
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of EmbeddingResults
        """
        await self.initialize()
        
        # Filter out empty texts
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)
        
        if not valid_texts:
            return []
        
        # Process in batches
        batch_size = self.config.embedding_batch_size
        results = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            batch_results = await self._provider.embed_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def embed_chunk(self, chunk) -> EmbeddingResult:
        """Embed a chunk with multi-vector strategy if enabled.
        
        Args:
            chunk: ChunkResult object with content and metadata
            
        Returns:
            EmbeddingResult
        """
        if self.config.use_multi_vector and chunk.metadata:
            text = self.create_multi_vector_text(chunk)
        else:
            text = chunk.content
        
        result = await self.embed_text(text, chunk.metadata)
        return result
    
    def create_multi_vector_text(self, chunk) -> str:
        """Create combined text for multi-vector strategy.
        
        Args:
            chunk: ChunkResult with content and metadata
            
        Returns:
            Combined text including content, summary, and questions
        """
        parts = [f"Content: {chunk.content}"]
        
        metadata = chunk.metadata
        if metadata.get("summary"):
            parts.append(f"Summary: {metadata['summary']}")
        
        if metadata.get("questions"):
            questions = metadata["questions"]
            if isinstance(questions, list):
                questions_text = " ".join(questions)
            else:
                questions_text = str(questions)
            parts.append(f"Questions: {questions_text}")
        
        if metadata.get("keywords"):
            keywords = metadata["keywords"]
            if isinstance(keywords, list):
                keywords_text = ", ".join(keywords)
            else:
                keywords_text = str(keywords)
            parts.append(f"Keywords: {keywords_text}")
        
        return "\n\n".join(parts)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.config.embedding_provider}:{self.config.embedding_model}:{text}".encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Embedding cache cleared")


# Provider implementations

class EmbeddingProvider:
    """Base class for embedding providers."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    async def initialize(self):
        """Initialize the provider."""
        pass
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text."""
        raise NotImplementedError
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts."""
        # Default implementation - providers can override for efficiency
        results = []
        for text in texts:
            result = await self.embed(text)
            results.append(result)
        return results
    
    def _normalize(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return (np.array(embedding) / norm).tolist()
        return embedding


class DashScopeProvider(EmbeddingProvider):
    """DashScope (Alibaba) embedding provider."""
    
    async def initialize(self):
        """Initialize DashScope client."""
        if not self.config.embedding_api_key:
            raise ValueError("DashScope API key is required")
        
        # In real implementation, would initialize client
        logger.info("DashScope provider initialized")
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed using DashScope API."""
        # This is a placeholder - real implementation would call API
        import random
        
        # Simulate API call
        await asyncio.sleep(0.01)  # Simulate network delay
        
        # Generate mock embedding
        embedding = [random.random() for _ in range(self.config.embedding_dimensions)]
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.config.embedding_model,
            usage={"total_tokens": len(text.split())}
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Batch embed using DashScope API."""
        # DashScope supports batch embedding
        # This would be more efficient than the default implementation
        results = []
        for text in texts:
            result = await self.embed(text)
            results.append(result)
        return results


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    async def initialize(self):
        """Initialize OpenAI client."""
        if not self.config.embedding_api_key:
            raise ValueError("OpenAI API key is required")
        
        logger.info("OpenAI provider initialized")
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed using OpenAI API."""
        # Placeholder implementation
        import random
        
        await asyncio.sleep(0.01)
        
        embedding = [random.random() for _ in range(self.config.embedding_dimensions)]
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.config.embedding_model,
            usage={"total_tokens": len(text.split())}
        )


class HuggingFaceProvider(EmbeddingProvider):
    """HuggingFace local embedding provider."""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self._model = None
    
    async def initialize(self):
        """Load HuggingFace model."""
        # In real implementation:
        # from sentence_transformers import SentenceTransformer
        # self._model = SentenceTransformer(self.config.embedding_model)
        logger.info(f"HuggingFace provider initialized with model: {self.config.embedding_model}")
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed using local model."""
        # Placeholder - real implementation would use the model
        import random
        
        # No network delay for local model
        embedding = [random.random() for _ in range(self.config.embedding_dimensions)]
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.config.embedding_model,
            usage={"total_tokens": 0}  # No token cost for local model
        )