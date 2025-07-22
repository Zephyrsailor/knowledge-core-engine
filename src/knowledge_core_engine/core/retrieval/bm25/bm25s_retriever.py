"""BM25S implementation - fast and lightweight BM25."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseBM25Retriever, BM25Result

logger = logging.getLogger(__name__)


class BM25SRetriever(BaseBM25Retriever):
    """BM25 retriever using the BM25S library.
    
    BM25S is a fast, lightweight implementation that uses sparse matrices
    for efficient computation.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        language: str = "en",
        persist_directory: str = "./data/bm25_index"
    ):
        """Initialize BM25S retriever.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            epsilon: Floor value for IDF
            language: Language for tokenization (en, zh, multi)
            persist_directory: Directory to save/load BM25 index
        """
        super().__init__()
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.language = language
        self.persist_directory = persist_directory
        self._retriever = None
        self._corpus_tokens = None
    
    async def _initialize(self) -> None:
        """Initialize BM25S."""
        try:
            import bm25s
            self._bm25s = bm25s
            logger.info("BM25S initialized successfully")
            
            # Try to load existing index
            await self._try_load_index()
            
        except ImportError:
            raise RuntimeError(
                "BM25S not installed. Please install with: pip install bm25s"
            )
    
    async def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the BM25S index."""
        if not self._initialized:
            await self.initialize()
        
        if not documents:
            return
        
        # Generate doc IDs if not provided
        if doc_ids is None:
            start_idx = len(self._documents)
            doc_ids = self._generate_doc_ids(len(documents), start_idx)
        
        # Ensure metadata list matches documents
        if metadata is None:
            metadata = [{} for _ in documents]
        elif isinstance(metadata, list) and len(metadata) == 1 and len(documents) > 1:
            # Handle case where single metadata is passed for multiple documents
            metadata = metadata * len(documents)
        
        # Store documents and metadata
        self._documents.extend(documents)
        self._doc_ids.extend(doc_ids)
        self._metadata.extend(metadata)
        
        # Rebuild index with all documents
        await self._rebuild_index()
        
        # Auto-save after adding documents
        await self._auto_save()
        
        logger.info(f"Added {len(documents)} documents to BM25S index, total documents: {len(self._documents)}")
    
    async def _rebuild_index(self) -> None:
        """Rebuild the BM25S index with all documents."""
        if not self._documents:
            return
        
        logger.info(f"Building BM25S index with {len(self._documents)} documents")
        
        # Tokenize documents
        if self.language == "zh":
            # For Chinese, use jieba for tokenization
            import jieba
            # Tokenize each document with jieba
            self._corpus_tokens = []
            for doc in self._documents:
                # Use jieba to tokenize directly
                tokens = list(jieba.cut(doc))
                self._corpus_tokens.append(tokens)
        else:
            # For English and other languages
            self._corpus_tokens = self._bm25s.tokenize(
                self._documents,
                stopwords="en" if self.language == "en" else None,
                stemmer=None,  # Avoid stemmer issues
                show_progress=False
            )
        
        # Create and index the retriever
        self._retriever = self._bm25s.BM25(
            k1=self.k1,
            b=self.b
        )
        self._retriever.index(self._corpus_tokens)
        
        logger.info("BM25S index built successfully")
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[BM25Result]:
        """Search for relevant documents using BM25S."""
        if not self._initialized:
            await self.initialize()
        
        if not self._retriever or not self._documents:
            return []
        
        # Tokenize query
        if self.language == "zh":
            # Use jieba for Chinese query tokenization
            import jieba
            query_tokens = [list(jieba.cut(query))]  # BM25S expects list of token lists
        else:
            query_tokens = self._bm25s.tokenize(
                query,
                stopwords="en" if self.language == "en" else None,
                stemmer=None,  # Avoid stemmer issues
                show_progress=False
            )
        
        # Retrieve documents
        doc_indices, scores = self._retriever.retrieve(
            query_tokens,
            k=min(top_k, len(self._documents))
        )
        
        # Handle different return types from bm25s
        if isinstance(doc_indices, np.ndarray):
            doc_indices = doc_indices.flatten().tolist()
        if isinstance(scores, np.ndarray):
            scores = scores.flatten().tolist()
        
        # Create results
        results = []
        for idx, score in zip(doc_indices, scores):
            # Apply metadata filter if provided
            if filter_metadata:
                doc_meta = self._metadata[idx]
                if not self._matches_filter(doc_meta, filter_metadata):
                    continue
            
            results.append(BM25Result(
                document_id=self._doc_ids[idx],
                document=self._documents[idx],
                score=float(score),
                metadata=self._metadata[idx].copy()
            ))
        
        # Sort by score (descending) and limit to top_k
        results.sort(reverse=True)
        return results[:top_k]
    
    async def delete_documents(self, doc_ids: List[str]) -> int:
        """Delete documents by IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not doc_ids:
            return 0
        
        # Create a set for faster lookup
        ids_to_delete = set(doc_ids)
        
        # Find indices to delete
        indices_to_delete = []
        for i, doc_id in enumerate(self._doc_ids):
            if doc_id in ids_to_delete:
                indices_to_delete.append(i)
        
        if not indices_to_delete:
            return 0
        
        # Sort indices in reverse order for safe deletion
        indices_to_delete.sort(reverse=True)
        
        # Delete from all lists
        for idx in indices_to_delete:
            del self._documents[idx]
            del self._doc_ids[idx]
            del self._metadata[idx]
        
        # Rebuild index
        if self._documents:
            await self._rebuild_index()
        else:
            self._retriever = None
            self._corpus_tokens = None
        
        # Auto-save after deletion
        await self._auto_save()
        
        deleted_count = len(indices_to_delete)
        logger.info(f"Deleted {deleted_count} documents from BM25S index")
        return deleted_count
    
    async def clear(self) -> None:
        """Clear all documents from the index."""
        self._documents = []
        self._doc_ids = []
        self._metadata = []
        self._retriever = None
        self._corpus_tokens = None
        logger.info("BM25S index cleared")
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_spec: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter specification.
        
        Supports simple equality and MongoDB-style operators:
        - {'key': value} - exact match
        - {'key': {'$gt': value}} - greater than
        - {'key': {'$gte': value}} - greater than or equal
        - {'key': {'$lt': value}} - less than
        - {'key': {'$lte': value}} - less than or equal
        - {'key': {'$in': [values]}} - value in list
        - {'key': {'$nin': [values]}} - value not in list
        """
        for key, expected in filter_spec.items():
            if key not in metadata:
                return False
            
            actual = metadata[key]
            
            # Simple equality check
            if not isinstance(expected, dict):
                if actual != expected:
                    return False
                continue
            
            # Complex operators
            for operator, op_value in expected.items():
                if operator == '$gt':
                    if not (actual > op_value):
                        return False
                elif operator == '$gte':
                    if not (actual >= op_value):
                        return False
                elif operator == '$lt':
                    if not (actual < op_value):
                        return False
                elif operator == '$lte':
                    if not (actual <= op_value):
                        return False
                elif operator == '$in':
                    # Handle both single values and lists
                    if isinstance(actual, list):
                        # Check if any element in actual list is in op_value
                        if not any(item in op_value for item in actual):
                            return False
                    else:
                        if actual not in op_value:
                            return False
                elif operator == '$nin':
                    if actual in op_value:
                        return False
                else:
                    # Unknown operator, treat as equality
                    if actual != expected:
                        return False
        
        return True
    
    async def save(self, path: str) -> None:
        """Save the BM25S index to disk."""
        if not self._retriever:
            raise ValueError("No index to save")
        
        import pickle
        import os
        
        # Create directory if needed
        os.makedirs(path, exist_ok=True)
        
        # Save BM25S model
        self._retriever.save(path)
        
        # Save additional data
        data = {
            "documents": self._documents,
            "doc_ids": self._doc_ids,
            "metadata": self._metadata,
            "corpus_tokens": self._corpus_tokens,
            "config": {
                "k1": self.k1,
                "b": self.b,
                "epsilon": self.epsilon,
                "language": self.language
            }
        }
        
        with open(os.path.join(path, "bm25s_data.pkl"), "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"BM25S index saved to {path}")
    
    async def load(self, path: str) -> None:
        """Load the BM25S index from disk."""
        if not self._initialized:
            await self.initialize()
        
        import pickle
        import os
        
        # Load BM25S model
        self._retriever = self._bm25s.BM25.load(path, load_corpus=True)
        
        # Load additional data
        with open(os.path.join(path, "bm25s_data.pkl"), "rb") as f:
            data = pickle.load(f)
        
        self._documents = data["documents"]
        self._doc_ids = data["doc_ids"]
        self._metadata = data["metadata"]
        self._corpus_tokens = data["corpus_tokens"]
        
        # Update config
        config = data["config"]
        self.k1 = config["k1"]
        self.b = config["b"]
        self.epsilon = config["epsilon"]
        self.language = config["language"]
        
        logger.info(f"BM25S index loaded from {path}")
    
    async def _try_load_index(self) -> None:
        """Try to load existing index from disk."""
        # Skip loading if no persist directory is configured
        if not self.persist_directory:
            logger.info("Persistence disabled, starting with empty index")
            return
            
        try:
            import os
            if os.path.exists(self.persist_directory):
                await self.load(self.persist_directory)
                logger.info(f"Loaded existing BM25 index with {len(self._documents)} documents")
        except Exception as e:
            logger.info(f"No existing BM25 index found or failed to load: {e}")
            # Continue with empty index
    
    async def _auto_save(self) -> None:
        """Auto-save the index after changes."""
        # Skip saving if no persist directory is configured
        if not self.persist_directory:
            return
            
        try:
            if self._retriever and len(self._documents) > 0:
                await self.save(self.persist_directory)
                logger.debug(f"BM25 index auto-saved to {self.persist_directory}")
        except Exception as e:
            logger.warning(f"Failed to auto-save BM25 index: {e}")