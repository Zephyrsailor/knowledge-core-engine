"""Base classes for reranker implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""
    
    document: str
    score: float
    index: int
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


class BaseReranker(ABC):
    """Abstract base class for all reranker implementations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize reranker with configuration.
        
        Args:
            config: Configuration dictionary for the reranker
        """
        self.config = config or {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the reranker.
        
        This method should be called before using the reranker.
        Subclasses should implement _initialize() for actual initialization.
        """
        if not self._initialized:
            await self._initialize()
            self._initialized = True
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize the reranker implementation.
        
        This method should load models, set up connections, etc.
        """
        pass
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True
    ) -> List[RerankResult]:
        """Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return (None = return all)
            return_documents: Whether to include full documents in results
            
        Returns:
            List of RerankResult objects sorted by relevance (highest first)
        """
        pass
    
    async def close(self) -> None:
        """Close any resources used by the reranker.
        
        Subclasses should override this method if they need to clean up resources.
        """
        pass
    
    async def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: Optional[int] = None
    ) -> List[List[RerankResult]]:
        """Batch rerank multiple query-documents pairs.
        
        Default implementation processes queries sequentially.
        Subclasses can override for parallel processing.
        
        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            top_k: Number of top results per query
            
        Returns:
            List of rerank results for each query
        """
        results = []
        for query, documents in zip(queries, documents_list):
            result = await self.rerank(query, documents, top_k)
            results.append(result)
        return results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range.
        
        Args:
            scores: Raw scores from the model
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
    
    async def close(self) -> None:
        """Clean up resources.
        
        Subclasses should implement _close() for actual cleanup.
        """
        if self._initialized:
            await self._close()
            self._initialized = False
    
    async def _close(self) -> None:
        """Clean up implementation-specific resources."""
        pass