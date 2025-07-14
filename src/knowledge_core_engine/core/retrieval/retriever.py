"""Main retriever implementation with multiple strategies."""

import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from ..config import RAGConfig
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Supported retrieval strategies."""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = None
    rerank_score: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def final_score(self) -> float:
        """Get final score (prefer rerank score if available)."""
        return self.rerank_score if self.rerank_score is not None else self.score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }


class Retriever:
    """Main retriever with support for multiple strategies."""
    
    def __init__(self, config: RAGConfig):
        """Initialize retriever with configuration.
        
        Args:
            config: RAG configuration object
        """
        self.config = config
        self._embedder = None
        self._vector_store = None
        self._bm25_index = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize retrieval components."""
        if self._initialized:
            return
        
        # Initialize embedder and vector store
        self._embedder = TextEmbedder(self.config)
        self._vector_store = VectorStore(self.config)
        
        await self._embedder.initialize()
        await self._vector_store.initialize()
        
        # Initialize BM25 if needed
        if self.config.retrieval_strategy in ["bm25", "hybrid"]:
            await self._initialize_bm25()
        
        self._initialized = True
        logger.info(f"Initialized retriever with strategy: {self.config.retrieval_strategy}")
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of retrieval results
        """
        await self.initialize()
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if top_k is None:
            top_k = self.config.retrieval_top_k
        
        # Apply query expansion if enabled
        if self.config.enable_query_expansion:
            expanded_queries = await self._expand_query(query)
            if len(expanded_queries) > 1:
                # Combine expanded queries
                query = " ".join(expanded_queries)
        
        # Route to appropriate strategy
        strategy = RetrievalStrategy(self.config.retrieval_strategy)
        
        if strategy == RetrievalStrategy.VECTOR:
            results = await self._vector_retrieve(query, top_k, filters)
        elif strategy == RetrievalStrategy.BM25:
            results = await self._bm25_retrieve(query, top_k, filters)
        elif strategy == RetrievalStrategy.HYBRID:
            results = await self._hybrid_retrieve(query, top_k, filters)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        
        return results
    
    async def _vector_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using vector similarity."""
        # Embed query
        embedding_result = await self._embedder.embed_text(query)
        
        # Search vector store
        query_results = await self._vector_store.query(
            query_embedding=embedding_result.embedding,
            top_k=top_k,
            filter=filters
        )
        
        # Convert to RetrievalResult
        results = []
        for qr in query_results:
            results.append(RetrievalResult(
                chunk_id=qr.id,
                content=qr.text,
                score=qr.score,
                metadata=qr.metadata
            ))
        
        return results
    
    async def _bm25_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using BM25."""
        # This would use the BM25 index
        # For now, return empty as BM25 is not implemented
        return []
    
    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 search."""
        # Placeholder for BM25 implementation
        # In real implementation, would use a BM25 index (e.g., Whoosh, Elasticsearch)
        return []
    
    async def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using hybrid strategy (vector + BM25)."""
        # Get weights
        vector_weight = self.config.vector_weight
        bm25_weight = self.config.bm25_weight
        
        # Perform both searches in parallel
        vector_task = self._vector_retrieve(query, top_k * 2, filters)
        bm25_task = self._bm25_retrieve(query, top_k * 2, filters)
        
        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
        
        # Combine results
        combined = self._combine_results(
            vector_results,
            bm25_results,
            vector_weight,
            bm25_weight
        )
        
        # Return top k
        return combined[:top_k]
    
    def _combine_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        vector_weight: float,
        bm25_weight: float
    ) -> List[RetrievalResult]:
        """Combine results from multiple sources."""
        # Track all unique documents
        combined_dict = {}
        
        # Add vector results
        for result in vector_results:
            combined_dict[result.chunk_id] = result
            result.metadata["vector_score"] = result.score
            result.metadata["fusion_method"] = "weighted"
        
        # Add/merge BM25 results
        for result in bm25_results:
            if result.chunk_id in combined_dict:
                # Merge scores
                existing = combined_dict[result.chunk_id]
                existing.metadata["bm25_score"] = result.score
                
                # Weighted combination
                existing.score = (
                    existing.metadata["vector_score"] * vector_weight +
                    result.score * bm25_weight
                )
            else:
                # New result from BM25
                result.metadata["bm25_score"] = result.score
                result.metadata["fusion_method"] = "weighted"
                result.score = result.score * bm25_weight
                combined_dict[result.chunk_id] = result
        
        # Sort by combined score
        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
    
    async def _expand_query(self, query: str) -> List[str]:
        """Expand query for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries (including original)
        """
        expanded = [query]  # Always include original
        
        if self.config.query_expansion_method == "llm":
            # Use LLM to generate related queries
            try:
                from ..generation.providers import create_llm_provider
                
                # Create a temporary LLM client for query expansion
                llm_provider = await create_llm_provider(self.config)
                
                # Build prompt for query expansion
                prompt = f"""为以下搜索查询生成{self.config.query_expansion_count - 1}个相关的搜索词或短语，
这些词应该：
1. 与原始查询相关但使用不同的表达方式
2. 包含同义词或相关概念
3. 帮助检索更多相关文档

原始查询：{query}

请直接返回相关查询，每行一个，不要编号或其他格式："""
                
                # Generate expanded queries
                messages = [{"role": "user", "content": prompt}]
                response = await llm_provider.generate(
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more focused expansion
                    max_tokens=200
                )
                
                # Parse response
                if response and "content" in response:
                    lines = response["content"].strip().split("\n")
                    for line in lines[:self.config.query_expansion_count - 1]:
                        line = line.strip()
                        if line and not line[0].isdigit():  # Skip numbered items
                            expanded.append(line)
                
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}, using original query only")
        
        elif self.config.query_expansion_method == "rule_based":
            # Simple rule-based expansion
            # Add common synonyms and variations
            import jieba
            
            # Tokenize query
            tokens = list(jieba.cut(query))
            
            # Simple synonym mapping (in production, use a proper synonym dictionary)
            synonyms = {
                "什么": ["哪些", "何种"],
                "如何": ["怎么", "怎样"],
                "为什么": ["为何", "原因"],
                "RAG": ["检索增强生成", "Retrieval Augmented Generation"],
                "技术": ["方法", "技巧"],
                "优势": ["优点", "好处", "长处"],
                "问题": ["挑战", "困难", "难点"]
            }
            
            # Generate variations
            for token in tokens:
                if token in synonyms:
                    for synonym in synonyms[token][:self.config.query_expansion_count - 1]:
                        variation = query.replace(token, synonym)
                        if variation != query and variation not in expanded:
                            expanded.append(variation)
                            if len(expanded) >= self.config.query_expansion_count:
                                break
                
                if len(expanded) >= self.config.query_expansion_count:
                    break
        
        logger.info(f"Query expanded from '{query}' to {len(expanded)} variations")
        return expanded
    
    def _normalize_score(self, score: float, source: str) -> float:
        """Normalize scores from different sources."""
        if source == "vector":
            # Vector scores are typically 0-1
            return min(max(score, 0.0), 1.0)
        elif source == "bm25":
            # BM25 scores can be > 1, normalize to 0-1
            # Simple sigmoid normalization
            import math
            return 1 / (1 + math.exp(-score / 10))
        else:
            return score
    
    async def _initialize_bm25(self):
        """Initialize BM25 index."""
        # Placeholder for BM25 initialization
        logger.info("BM25 index initialization placeholder")
    
    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[List[RetrievalResult]]:
        """Retrieve for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            filters: Optional metadata filters
            
        Returns:
            List of result lists, one per query
        """
        results = []
        
        for query in queries:
            try:
                query_results = await self.retrieve(query, top_k, filters)
                results.append(query_results)
            except Exception as e:
                logger.error(f"Error retrieving for query '{query}': {e}")
                if self.config.extra_params.get("batch_ignore_errors", False):
                    results.append([])
                else:
                    raise
        
        return results