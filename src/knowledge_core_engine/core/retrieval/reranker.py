"""Reranker implementation with support for multiple providers."""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import numpy as np

from ..config import RAGConfig
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""
    index: int
    score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __gt__(self, other):
        """Compare by score for sorting."""
        return self.score > other.score
    
    def __eq__(self, other):
        """Check equality by score."""
        return self.score == other.score


class Reranker:
    """Reranker with support for multiple providers."""
    
    def __init__(self, config: RAGConfig):
        """Initialize reranker with configuration.
        
        Args:
            config: RAG configuration object
        """
        self.config = config
        self._provider = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the reranker provider."""
        if self._initialized:
            return
        
        # Create provider based on config
        if self.config.reranker_provider == "huggingface":
            if "bge" in self.config.reranker_model.lower():
                self._provider = BGERerankerProvider(self.config)
            else:
                raise ValueError(f"Unknown HuggingFace reranker: {self.config.reranker_model}")
        elif self.config.reranker_provider == "cohere":
            self._provider = CohereRerankerProvider(self.config)
        else:
            # Default to BGE if not specified
            self._provider = BGERerankerProvider(self.config)
        
        await self._provider.initialize()
        self._initialized = True
        
        logger.info(f"Initialized reranker: {self.config.reranker_model}")
    
    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Rerank retrieval results.
        
        Args:
            query: Original query
            results: List of retrieval results to rerank
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked results
        """
        if not results:
            return []
        
        if len(results) == 1:
            # No need to rerank single result
            results[0].rerank_score = 0.92  # Default high score
            return results
        
        await self.initialize()
        
        if top_k is None:
            top_k = self.config.extra_params.get("rerank_top_k", len(results))
        
        # Extract texts for reranking
        texts = [r.content for r in results]
        
        # Get rerank scores
        rerank_results = await self._provider.rerank(
            query=query,
            texts=texts,
            batch_size=self.config.extra_params.get("rerank_batch_size", 32)
        )
        
        # Apply scores to results
        reranked = []
        for rr in rerank_results:
            if rr.index < len(results):
                result = results[rr.index]
                result.rerank_score = rr.score
                result.metadata["original_rank"] = rr.index + 1
                reranked.append((rr.score, result))
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x[0], reverse=True)
        
        # Apply score threshold if configured
        threshold = self.config.extra_params.get("rerank_score_threshold")
        if threshold:
            reranked = [(s, r) for s, r in reranked if s >= threshold]
        
        # Return top k
        final_results = [r for _, r in reranked[:top_k]]
        
        return final_results


# Provider implementations

class RerankerProvider:
    """Base class for reranker providers."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    async def initialize(self):
        """Initialize the provider."""
        pass
    
    async def rerank(
        self,
        query: str,
        texts: List[str],
        batch_size: int = 32
    ) -> List[RerankResult]:
        """Rerank texts based on query relevance."""
        raise NotImplementedError


class BGERerankerProvider(RerankerProvider):
    """BGE reranker provider using HuggingFace models."""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self._model = None
    
    async def initialize(self):
        """Load BGE reranker model."""
        # In real implementation:
        # from sentence_transformers import CrossEncoder
        # self._model = CrossEncoder(self.config.reranker_model)
        logger.info(f"BGE reranker initialized: {self.config.reranker_model}")
    
    async def rerank(
        self,
        query: str,
        texts: List[str],
        batch_size: int = 32
    ) -> List[RerankResult]:
        """Rerank using BGE model."""
        # Placeholder implementation
        # Real implementation would use the model
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_scores = await self._model_predict(query, batch_texts)
            
            for j, score in enumerate(batch_scores):
                results.append(RerankResult(
                    index=i + j,
                    score=score
                ))
        
        return results
    
    async def _model_predict(self, query: str, texts: List[str]) -> List[float]:
        """Get model predictions."""
        # Placeholder - real implementation would call model
        # Simulate scoring
        import random
        scores = []
        for text in texts:
            # Simulate relevance scoring
            base_score = random.random()
            # Boost if query terms in text
            query_terms = set(query.lower().split())
            text_terms = set(text.lower().split())
            overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
            score = base_score * 0.5 + overlap * 0.5
            scores.append(score)
        
        return scores
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding


class CohereRerankerProvider(RerankerProvider):
    """Cohere reranker provider."""
    
    async def initialize(self):
        """Initialize Cohere client."""
        if not self.config.reranker_api_key:
            raise ValueError("Cohere API key required for reranker")
        
        logger.info(f"Cohere reranker initialized: {self.config.reranker_model}")
    
    async def rerank(
        self,
        query: str,
        texts: List[str],
        batch_size: int = 32
    ) -> List[RerankResult]:
        """Rerank using Cohere API."""
        # Placeholder implementation
        # Real implementation would call Cohere API
        
        # Simulate API response
        results = []
        for i, text in enumerate(texts):
            score = 0.5 + (len(set(query.split()) & set(text.split())) * 0.1)
            results.append(RerankResult(
                index=i,
                score=min(score, 1.0)
            ))
        
        return results