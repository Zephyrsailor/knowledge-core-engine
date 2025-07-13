"""Embedding module for converting text to vectors.

This module provides:
- Flexible embedding providers (DashScope, OpenAI, HuggingFace, etc.)
- Configurable embedding strategies
- Vector storage abstraction (ChromaDB, Pinecone, Weaviate, etc.)
- Easy integration as a standalone module

Example:
    ```python
    from knowledge_core_engine.core.config import RAGConfig
    from knowledge_core_engine.core.embedding import TextEmbedder, VectorStore
    
    # Configure
    config = RAGConfig(
        embedding_provider="dashscope",
        vectordb_provider="chromadb"
    )
    
    # Create embedder
    embedder = TextEmbedder(config)
    
    # Embed text
    result = await embedder.embed_text("Your text here")
    
    # Store in vector database
    vector_store = VectorStore(config)
    await vector_store.add_document(result)
    ```
"""

from .embedder import (
    TextEmbedder,
    EmbeddingResult,
    EmbeddingProvider
)
from .vector_store import (
    VectorStore,
    VectorDocument,
    QueryResult,
    VectorStoreProvider
)

__all__ = [
    # Embedder
    "TextEmbedder",
    "EmbeddingResult",
    "EmbeddingProvider",
    
    # Vector Store
    "VectorStore",
    "VectorDocument", 
    "QueryResult",
    "VectorStoreProvider",
]