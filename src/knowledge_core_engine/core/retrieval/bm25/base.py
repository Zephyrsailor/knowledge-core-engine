"""Base classes for BM25 implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """Result from BM25 search."""
    
    document_id: str
    document: str
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


class BaseBM25Retriever(ABC):
    """Abstract base class for BM25 retrievers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize BM25 retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._initialized = False
        self._documents: List[str] = []
        self._doc_ids: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
    
    async def initialize(self) -> None:
        """Initialize the retriever."""
        if not self._initialized:
            await self._initialize()
            self._initialized = True
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize the implementation."""
        pass
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the index.
        
        Args:
            documents: List of document texts
            doc_ids: Optional document IDs (auto-generated if not provided)
            metadata: Optional metadata for each document
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[BM25Result]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of BM25Result objects sorted by relevance
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all documents from the index."""
        pass
    
    async def save(self, path: str) -> None:
        """Save the index to disk.
        
        Args:
            path: Path to save the index
        """
        raise NotImplementedError("Save functionality not implemented")
    
    async def load(self, path: str) -> None:
        """Load the index from disk.
        
        Args:
            path: Path to load the index from
        """
        raise NotImplementedError("Load functionality not implemented")
    
    def _generate_doc_ids(self, num_docs: int, start_idx: int = 0) -> List[str]:
        """Generate document IDs.
        
        Args:
            num_docs: Number of document IDs to generate
            start_idx: Starting index
            
        Returns:
            List of document IDs
        """
        return [f"doc_{start_idx + i}" for i in range(num_docs)]
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._initialized:
            await self._close()
            self._initialized = False
    
    async def _close(self) -> None:
        """Clean up implementation-specific resources."""
        pass