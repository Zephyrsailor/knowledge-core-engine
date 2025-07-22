"""BM25 retrieval integration with new provider system."""

import logging
from typing import List, Dict, Any, Optional

from ..config import RAGConfig
from .bm25.factory import create_bm25_retriever
from .bm25.base import BaseBM25Retriever, BM25Result

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 retriever wrapper that uses the new provider system."""
    
    def __init__(self, config: Optional[RAGConfig] = None, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """Initialize BM25 retriever with configuration or parameters.
        
        Args:
            config: RAG configuration object (if None, will create default config)
            k1: BM25 k1 parameter (for backward compatibility)
            b: BM25 b parameter (for backward compatibility) 
            epsilon: BM25 epsilon parameter (for backward compatibility)
        """
        # Handle backward compatibility
        if config is None:
            # Create a minimal config for direct parameter usage
            from ..config import RAGConfig
            self.config = RAGConfig(
                embedding_provider="mock",
                embedding_model="mock",
                bm25_provider="bm25s",
                retrieval_strategy="vector"
            )
            # Store parameters for later use
            self._k1 = k1
            self._b = b
            self._epsilon = epsilon
        else:
            self.config = config
            self._k1 = k1
            self._b = b
            self._epsilon = epsilon
            
        self._retriever: Optional[BaseBM25Retriever] = None
        self._initialized = False
        
        # Document tracking for compatibility
        self.documents: Dict[str, str] = {}
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}
        self.doc_ids: List[str] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        
        # BM25 parameters for compatibility
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
    
    async def initialize(self):
        """Initialize the BM25 provider."""
        if self._initialized:
            return
        
        # For backward compatibility, create BM25S retriever directly if no config provider
        if self.config.bm25_provider == "bm25s" or not hasattr(self.config, 'bm25_provider'):
            from .bm25.bm25s_retriever import BM25SRetriever
            self._retriever = BM25SRetriever(
                k1=self._k1,
                b=self._b,
                epsilon=self._epsilon,
                language="zh",  # Default to Chinese for now
                persist_directory=None  # Disable persistence for testing
            )
        else:
            # Create BM25 retriever using factory
            self._retriever = create_bm25_retriever(self.config)
        
        if self._retriever:
            await self._retriever.initialize()
            logger.info(f"Initialized BM25 retriever with k1={self.k1}, b={self.b}")
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
            if doc_id not in self.doc_ids:
                self.doc_ids.append(doc_id)
            
            # Prepare for BM25
            doc_texts.append(content)
            doc_ids.append(doc_id)
            doc_metadata.append(metadata)
        
        # Add to BM25 index - 确保传递正确的参数格式
        if doc_texts:
            await self._retriever.add_documents(
                documents=doc_texts,
                doc_ids=doc_ids,
                metadata=doc_metadata
            )
        
        # Update statistics for compatibility
        self._update_stats()
        
        logger.info(f"Added {len(documents)} documents to BM25 index")
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
            return loop.run_until_complete(self._search_async(query, top_k, filters))
        finally:
            loop.close()
    
    async def _search_async(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using BM25 (async version)."""
        await self.initialize()
        
        if not self._retriever:
            logger.warning("BM25 retriever not available, returning empty results")
            return []
        
        # Perform search
        results = await self._retriever.search(query, top_k, filter_metadata=filters)
        
        # Convert to expected format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.document_id,
                "content": result.document,
                "score": result.score,
                "metadata": result.metadata
            })
        
        logger.info(f"BM25 search returned {len(formatted_results)} results for query: {query[:50]}...")
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
        self.doc_ids.clear()
        self.idf.clear()
        self.avgdl = 0.0
        
        logger.info("BM25 index cleared")
    
    def update_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update a document in the index."""
        import asyncio
        
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._update_document_async(doc_id, content, metadata))
        finally:
            loop.close()
    
    async def _update_document_async(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update a document in the index (async version)."""
        if doc_id in self.documents:
            # Remove old document
            await self._remove_document_async(doc_id)
        
        # Add updated document
        doc = {
            "id": doc_id,
            "content": content,
            "metadata": metadata or {}
        }
        await self._add_documents_async([doc])
    
    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index."""
        import asyncio
        
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._remove_document_async(doc_id))
        finally:
            loop.close()
    
    async def _remove_document_async(self, doc_id: str) -> None:
        """Remove a document from the index (async version)."""
        if doc_id not in self.documents:
            return
        
        await self.initialize()
        
        if self._retriever:
            await self._retriever.delete_documents([doc_id])
        
        # Remove from compatibility structures
        if doc_id in self.documents:
            del self.documents[doc_id]
        if doc_id in self.doc_metadata:
            del self.doc_metadata[doc_id]
        if doc_id in self.doc_ids:
            self.doc_ids.remove(doc_id)
        
        self._update_stats()
        logger.info(f"Removed document {doc_id} from BM25 index")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "num_documents": len(self.documents),
            "num_terms": len(self.idf),
            "avg_doc_length": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon
        }
    
    def _update_stats(self) -> None:
        """Update internal statistics for compatibility."""
        if not self.documents:
            self.avgdl = 0.0
            self.idf.clear()
            return
        
        # Calculate average document length
        total_length = sum(len(doc.split()) for doc in self.documents.values())
        self.avgdl = total_length / len(self.documents)
        
        # Mock IDF calculation for compatibility
        # In real BM25, this would be calculated properly
        import math
        for doc in self.documents.values():
            words = doc.split()
            for word in set(words):
                if word not in self.idf:
                    # Simple mock IDF
                    doc_freq = sum(1 for d in self.documents.values() if word in d)
                    self.idf[word] = math.log((len(self.documents) + 1) / (doc_freq + 1))