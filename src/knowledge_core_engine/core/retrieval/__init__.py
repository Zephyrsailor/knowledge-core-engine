"""Retrieval module for K-Engine.

This module provides:
- Hybrid retrieval strategies (vector + BM25)
- Advanced reranking with BGE models
- Query optimization and expansion
- Multi-stage retrieval pipelines

Example:
    ```python
    from knowledge_core_engine.core.config import RAGConfig
    from knowledge_core_engine.core.retrieval import Retriever
    
    # Configure
    config = RAGConfig(
        retrieval_strategy="hybrid",
        reranker_model="bge-reranker-v2-m3-qwen"
    )
    
    # Create retriever
    retriever = Retriever(config)
    
    # Search
    results = await retriever.retrieve(
        query="什么是RAG技术？",
        top_k=5
    )
    ```
"""

from .retriever import (
    Retriever,
    RetrievalResult,
    RetrievalStrategy
)
from .reranker import (
    Reranker,
    RerankResult,
    RerankerProvider,
    BGERerankerProvider,
    CohereRerankerProvider
)
from .hybrid_strategy import (
    HybridRetriever,
    ScoreFusion,
    ReciprocalRankFusion
)

__all__ = [
    # Retriever
    "Retriever",
    "RetrievalResult",
    "RetrievalStrategy",
    
    # Reranker
    "Reranker",
    "RerankResult", 
    "RerankerProvider",
    "BGERerankerProvider",
    "CohereRerankerProvider",
    
    # Hybrid Strategy
    "HybridRetriever",
    "ScoreFusion",
    "ReciprocalRankFusion",
]