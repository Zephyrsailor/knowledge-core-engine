"""Reranker integration with new provider system."""

import logging
from typing import List, Optional

from ..config import RAGConfig
from .retriever import RetrievalResult
from .reranker.factory import create_reranker
from .reranker.base import BaseReranker

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker wrapper that uses the new provider system."""
    
    def __init__(self, config: RAGConfig):
        """Initialize reranker with configuration.
        
        Args:
            config: RAG configuration object
        """
        self.config = config
        self._reranker: Optional[BaseReranker] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the reranker provider."""
        if self._initialized:
            return
        
        # Create reranker using factory
        self._reranker = create_reranker(self.config)
        
        if self._reranker:
            await self._reranker.initialize()
            logger.info(
                f"Initialized reranker: provider={self.config.reranker_provider}, "
                f"model={self.config.reranker_model}"
            )
        else:
            logger.info("Reranking is disabled")
        
        self._initialized = True
    
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
        
        if not self.config.enable_reranking or not self._reranker:
            # Return original results if reranking is disabled
            return results[:top_k] if top_k else results
        
        if len(results) == 1:
            # No need to rerank single result
            results[0].rerank_score = 0.95
            return results
        
        await self.initialize()
        
        if top_k is None:
            top_k = self.config.rerank_top_k
        
        # Extract texts for reranking
        documents = [r.content for r in results]
        
        # Perform reranking
        rerank_results = await self._reranker.rerank(
            query=query,
            documents=documents,
            top_k=len(results),  # Get scores for all documents
            return_documents=False  # We already have the documents
        )
        
        # Map scores back to original results
        score_map = {r.index: r.score for r in rerank_results}
        
        # Update rerank scores
        for i, result in enumerate(results):
            if i in score_map:
                result.rerank_score = score_map[i]
            else:
                result.rerank_score = 0.0
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        
        # Return top_k results
        return results[:top_k]
    
    async def close(self):
        """Clean up resources."""
        if self._reranker:
            await self._reranker.close()
            self._reranker = None
            self._initialized = False