"""Embedding provider abstraction for flexible embedding integration."""

from abc import abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

from .base import Provider, ProviderConfig, ProviderFactory


@dataclass
class EmbeddingConfig(ProviderConfig):
    """Embedding-specific configuration."""
    model: str = None  # Model name
    dimensions: int = None  # Vector dimensions
    batch_size: int = 25
    normalize: bool = True


@dataclass
class EmbeddingResult:
    """Standard embedding result format."""
    text: str
    embedding: Union[List[float], np.ndarray]
    model: str
    usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}
    
    @property
    def dimension(self) -> int:
        return len(self.embedding)


class EmbeddingProvider(Provider):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with vector
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
        """Get the dimension of embeddings."""
        pass
    
    def prepare_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Prepare text for embedding (can be overridden).
        
        Default implementation for multi-vector strategy.
        """
        if not metadata:
            return text
        
        parts = [f"Content: {text}"]
        
        if metadata.get("summary"):
            parts.append(f"Summary: {metadata['summary']}")
        
        if metadata.get("questions"):
            questions = metadata["questions"]
            if isinstance(questions, list):
                parts.append(f"Questions: {' '.join(questions)}")
        
        return "\n\n".join(parts)


# --- DashScope Implementation ---

class DashScopeEmbeddingProvider(EmbeddingProvider):
    """DashScope (Alibaba Cloud) embedding provider."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model = config.model or "text-embedding-v3"
        self.dimensions = config.dimensions or 1536
        self.api_base = config.api_base or "https://dashscope.aliyuncs.com/api/v1"
    
    async def initialize(self):
        """Initialize DashScope client."""
        if not self.config.api_key:
            raise ValueError("DashScope API key is required")
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed using DashScope API."""
        # Placeholder implementation
        return EmbeddingResult(
            text=text,
            embedding=[0.1] * self.dimensions,
            model=self.model,
            usage={"tokens": len(text.split())}
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Batch embed using DashScope API."""
        # Process in batches according to config.batch_size
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            # API call would go here
            for text in batch:
                results.append(await self.embed(text))
        return results
    
    def get_dimension(self) -> int:
        return self.dimensions
    
    async def close(self):
        """Clean up resources."""
        pass


# --- OpenAI Implementation ---

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model = config.model or "text-embedding-3-large"
        self.dimensions = config.dimensions or 3072
        self.api_base = config.api_base or "https://api.openai.com/v1"
    
    async def initialize(self):
        """Initialize OpenAI client."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed using OpenAI API."""
        return EmbeddingResult(
            text=text,
            embedding=[0.2] * self.dimensions,
            model=self.model,
            usage={"tokens": len(text.split())}
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Batch embed using OpenAI API."""
        # OpenAI supports larger batches
        results = []
        for text in texts:
            results.append(await self.embed(text))
        return results
    
    def get_dimension(self) -> int:
        return self.dimensions
    
    async def close(self):
        """Clean up resources."""
        pass


# --- HuggingFace Local Implementation ---

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace local embedding provider."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model = config.model or "BAAI/bge-large-zh-v1.5"
        self.dimensions = config.dimensions or 1024
        self._model = None
    
    async def initialize(self):
        """Load HuggingFace model."""
        # In real implementation, would load the model
        # from sentence_transformers import SentenceTransformer
        # self._model = SentenceTransformer(self.model)
        pass
    
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed using local model."""
        # Placeholder
        return EmbeddingResult(
            text=text,
            embedding=[0.3] * self.dimensions,
            model=self.model,
            usage={"tokens": 0}  # Local model, no token cost
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Batch embed using local model."""
        results = []
        for text in texts:
            results.append(await self.embed(text))
        return results
    
    def get_dimension(self) -> int:
        return self.dimensions
    
    async def close(self):
        """Clean up resources."""
        self._model = None


# Register providers
ProviderFactory.register("embedding", "dashscope", DashScopeEmbeddingProvider)
ProviderFactory.register("embedding", "openai", OpenAIEmbeddingProvider)
ProviderFactory.register("embedding", "huggingface", HuggingFaceEmbeddingProvider)


# --- Usage Example ---
"""
# Configuration
config = {
    "provider": "dashscope",
    "api_key": "sk-xxx",  # or from env: DASHSCOPE_API_KEY
    "model": "text-embedding-v3",
    "dimensions": 1536,
    "batch_size": 25
}

# Create embedding provider
embedder = ProviderFactory.create("embedding", config)
await embedder.initialize()

# Simple embedding
result = await embedder.embed("What is RAG?")
print(f"Dimension: {result.dimension}")

# With metadata (multi-vector strategy)
text = "RAG combines retrieval and generation"
metadata = {
    "summary": "RAG技术结合检索和生成",
    "questions": ["什么是RAG?", "RAG如何工作?"]
}
prepared = embedder.prepare_text(text, metadata)
result = await embedder.embed(prepared)

# Batch embedding
texts = ["text1", "text2", "text3"]
results = await embedder.embed_batch(texts)
"""