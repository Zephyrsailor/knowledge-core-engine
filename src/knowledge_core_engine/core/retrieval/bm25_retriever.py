"""BM25 retrieval integration with new provider system."""

import logging
from typing import List, Dict, Any, Optional

from ..config import RAGConfig
from .bm25.factory import create_bm25_retriever
from .bm25.base import BaseBM25Retriever, BM25Result

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 retriever wrapper that uses the new provider system."""
    
    def __init__(self, config: RAGConfig):
        """Initialize BM25 retriever with configuration.
        
        Args:
            config: RAG configuration object
        """
        self.config = config
        self._retriever: Optional[BaseBM25Retriever] = None
        self._initialized = False
        
        # Document tracking for compatibility
        self.documents: Dict[str, str] = {}
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize the BM25 provider."""
        if self._initialized:
            return
        
        # Create BM25 retriever using factory
        self._retriever = create_bm25_retriever(self.config)
        
        if self._retriever:
            await self._retriever.initialize()
            logger.info(f"Initialized BM25 retriever: provider={self.config.bm25_provider}")
        else:
            logger.info("BM25 retrieval not needed for strategy: %s", self.config.retrieval_strategy)
        
        self._initialized = True
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the BM25 index.
        
        This method is synchronous for backward compatibility.
        It internally calls the async method.
        
        Args:
            documents: List of documents with 'id', 'content', and optional 'metadata'
        """
        import asyncio
        
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._add_documents_async(documents))
        finally:
            loop.close()
    
    async def _add_documents_async(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the BM25 index (async version)."""
        if not documents:
            return
        
        await self.initialize()
        
        if not self._retriever:
            logger.warning("BM25 retriever not available, skipping document addition")
            return
        
        # Extract document data
        doc_texts = []
        doc_ids = []
        doc_metadata = []
        
        for doc in documents:
            doc_id = doc.get("id", f"doc_{len(self.documents)}")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Store for compatibility
            self.documents[doc_id] = content
            self.doc_metadata[doc_id] = metadata
            
            # Prepare for BM25
            doc_texts.append(content)
            doc_ids.append(doc_id)
            doc_metadata.append(metadata)
        
        # Add to BM25 index
        await self._retriever.add_documents(doc_texts, doc_ids, doc_metadata)
        
        logger.info(f"Added {len(documents)} documents to BM25 index")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant documents using BM25.
        
        This method is synchronous for backward compatibility.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        import asyncio
        
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._search_async(query, top_k))
        finally:
            loop.close()
    
    async def _search_async(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant documents using BM25 (async version)."""
        await self.initialize()
        
        if not self._retriever:
            logger.warning("BM25 retriever not available, returning empty results")
            return []
        
        # Perform search
        results = await self._retriever.search(query, top_k)
        
        # Convert to expected format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.document_id,
                "content": result.document,
                "score": result.score,
                "metadata": result.metadata
            })
        
        return formatted_results
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        import asyncio
        
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._clear_async())
        finally:
            loop.close()
    
    async def _clear_async(self) -> None:
        """Clear all documents from the index (async version)."""
        if self._retriever:
            await self._retriever.clear()
        
        self.documents.clear()
        self.doc_metadata.clear()
        
        logger.info("BM25 index cleared")