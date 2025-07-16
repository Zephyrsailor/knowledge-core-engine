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
        language: str = "en"
    ):
        """Initialize BM25S retriever.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            epsilon: Floor value for IDF
            language: Language for tokenization (en, zh, multi)
        """
        super().__init__()
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.language = language
        self._retriever = None
        self._corpus_tokens = None
    
    async def _initialize(self) -> None:
        """Initialize BM25S."""
        try:
            import bm25s
            self._bm25s = bm25s
            logger.info("BM25S initialized successfully")
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
        
        # Store documents and metadata
        self._documents.extend(documents)
        self._doc_ids.extend(doc_ids)
        self._metadata.extend(metadata)
        
        # Rebuild index with all documents
        await self._rebuild_index()
    
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
                if not all(
                    doc_meta.get(k) == v 
                    for k, v in filter_metadata.items()
                ):
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
    
    async def clear(self) -> None:
        """Clear all documents from the index."""
        self._documents = []
        self._doc_ids = []
        self._metadata = []
        self._retriever = None
        self._corpus_tokens = None
        logger.info("BM25S index cleared")
    
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