"""Hybrid retrieval strategies and score fusion methods."""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import logging

from ..config import RAGConfig
from .retriever import RetrievalResult
from ..embedding.vector_store import QueryResult
from .bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval with multiple strategies."""
    
    def __init__(self, config: RAGConfig):
        """Initialize hybrid retriever.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self._vector_store = None
        self._bm25_index = None
        self._bm25_retriever = BM25Retriever(config)
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform hybrid retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of retrieval results
        """
        # Get results from both sources
        vector_results = await self._vector_search(query, top_k * 2, filters)
        bm25_results = await self._bm25_search(query, top_k * 2, filters)
        
        # Fuse results based on configured method
        fusion_method = self.config.fusion_method
        
        if fusion_method == "weighted":
            results = self._weighted_fusion(
                vector_results,
                bm25_results,
                self.config.vector_weight,
                self.config.bm25_weight
            )
        elif fusion_method == "rrf":
            rrf = ReciprocalRankFusion(k=60)
            results = rrf.fuse(vector_results, bm25_results)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        return results[:top_k]
    
    async def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Perform vector search."""
        # Placeholder - would use actual vector store
        return []
    
    async def _bm25_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 search."""
        # Placeholder - would use actual BM25 index
        return []
    
    def _normalize_scores(
        self,
        results: List[Union[QueryResult, Dict[str, Any]]],
        method: str = "min_max"
    ) -> List[Dict[str, Any]]:
        """Normalize scores to [0, 1] range.
        
        Args:
            results: List of results with scores
            method: Normalization method (min_max, z_score)
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return []
        
        # Extract scores
        if isinstance(results[0], QueryResult):
            scores = [r.score for r in results]
        else:
            scores = [r["score"] for r in results]
        
        if method == "min_max":
            min_score = min(scores)
            max_score = max(scores)
            range_score = max_score - min_score
            
            if range_score == 0:
                # All scores are the same
                normalized_scores = [1.0] * len(scores)
            else:
                normalized_scores = [(s - min_score) / range_score for s in scores]
        
        elif method == "z_score":
            mean = np.mean(scores)
            std = np.std(scores)
            
            if std == 0:
                normalized_scores = [0.0] * len(scores)
            else:
                normalized_scores = [(s - mean) / std for s in scores]
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Apply normalized scores
        normalized_results = []
        for i, result in enumerate(results):
            if isinstance(result, QueryResult):
                normalized_results.append({
                    "id": result.id,
                    "score": normalized_scores[i],
                    "text": result.text,
                    "metadata": result.metadata
                })
            else:
                result_copy = result.copy()
                result_copy["score"] = normalized_scores[i]
                normalized_results.append(result_copy)
        
        return normalized_results
    
    def _weighted_fusion(
        self,
        vector_results: List[Union[QueryResult, Dict[str, Any]]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float,
        bm25_weight: float
    ) -> List[RetrievalResult]:
        """Fuse results using weighted combination.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            vector_weight: Weight for vector scores
            bm25_weight: Weight for BM25 scores
            
        Returns:
            Fused results
        """
        # Normalize scores
        norm_method = "min_max"  # Default normalization method
        vector_norm = self._normalize_scores(vector_results, norm_method)
        bm25_norm = self._normalize_scores(bm25_results, norm_method)
        
        # Build result map
        result_map = {}
        
        # Add vector results
        for result in vector_norm:
            chunk_id = result["id"]
            result_map[chunk_id] = RetrievalResult(
                chunk_id=chunk_id,
                content=result["text"],
                score=result["score"] * vector_weight,
                metadata={
                    **result.get("metadata", {}),
                    "vector_score": result["score"],
                    "fusion_method": "weighted"
                }
            )
        
        # Add/merge BM25 results
        for result in bm25_norm:
            chunk_id = result["id"]
            
            if chunk_id in result_map:
                # Merge with existing
                existing = result_map[chunk_id]
                existing.metadata["bm25_score"] = result["score"]
                existing.score += result["score"] * bm25_weight
            else:
                # New result
                result_map[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    content=result["text"],
                    score=result["score"] * bm25_weight,
                    metadata={
                        **result.get("metadata", {}),
                        "bm25_score": result["score"],
                        "fusion_method": "weighted"
                    }
                )
        
        # Sort by fused score
        results = list(result_map.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining ranked lists."""
    
    def __init__(self, k: int = 60):
        """Initialize RRF.
        
        Args:
            k: RRF parameter (typically 60)
        """
        self.k = k
    
    def fuse(
        self,
        list1: List[Union[QueryResult, Dict[str, Any]]],
        list2: List[Union[QueryResult, Dict[str, Any]]]
    ) -> List[RetrievalResult]:
        """Fuse two ranked lists using RRF.
        
        Args:
            list1: First ranked list
            list2: Second ranked list
            
        Returns:
            Fused results
        """
        return self.fuse_multiple([list1, list2])
    
    def fuse_multiple(
        self,
        lists: List[List[Union[QueryResult, Dict[str, Any]]]]
    ) -> List[RetrievalResult]:
        """Fuse multiple ranked lists using RRF.
        
        Args:
            lists: List of ranked lists
            
        Returns:
            Fused results
        """
        # Calculate RRF scores
        rrf_scores = {}
        doc_info = {}
        
        for list_idx, ranked_list in enumerate(lists):
            for rank, item in enumerate(ranked_list):
                # Get document ID
                if isinstance(item, QueryResult):
                    doc_id = item.id
                    text = item.text
                    metadata = item.metadata
                else:
                    doc_id = item["id"]
                    text = item.get("text", "")  # text might be optional
                    metadata = item.get("metadata", {})
                
                # Calculate RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank + 1)  # rank is 0-indexed
                
                # Accumulate scores
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                    doc_info[doc_id] = (text, metadata)
                
                rrf_scores[doc_id] += rrf_score
        
        # Create results
        results = []
        for doc_id, score in rrf_scores.items():
            text, metadata = doc_info[doc_id]
            results.append(RetrievalResult(
                chunk_id=doc_id,
                content=text,
                score=score,
                metadata={
                    **metadata,
                    "fusion_method": "rrf",
                    "rrf_k": self.k
                }
            ))
        
        # Sort by RRF score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


class ScoreFusion:
    """Different score fusion strategies."""
    
    def __init__(self, method: str = "weighted", weights: Optional[Dict[str, float]] = None):
        """Initialize score fusion.
        
        Args:
            method: Fusion method (weighted, max, min, mean)
            weights: Weights for different sources (for weighted method)
        """
        self.method = method
        self.weights = weights or {"vector": 0.7, "bm25": 0.3}
    
    def fuse_scores(
        self,
        result1: RetrievalResult,
        result2: RetrievalResult
    ) -> RetrievalResult:
        """Fuse scores from two results for the same document.
        
        Args:
            result1: First result
            result2: Second result (same document)
            
        Returns:
            Fused result
        """
        if result1.chunk_id != result2.chunk_id:
            raise ValueError("Cannot fuse results from different documents")
        
        # Determine sources
        source1 = result1.metadata.get("source", "unknown")
        source2 = result2.metadata.get("source", "unknown")
        
        # Calculate fused score
        if self.method == "max":
            fused_score = max(result1.score, result2.score)
        elif self.method == "min":
            fused_score = min(result1.score, result2.score)
        elif self.method == "mean":
            fused_score = (result1.score + result2.score) / 2
        elif self.method == "weighted":
            weight1 = self.weights.get(source1, 0.5)
            weight2 = self.weights.get(source2, 0.5)
            # Normalize weights
            total_weight = weight1 + weight2
            if total_weight > 0:
                weight1 /= total_weight
                weight2 /= total_weight
            fused_score = result1.score * weight1 + result2.score * weight2
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
        
        # Create fused result
        fused_result = RetrievalResult(
            chunk_id=result1.chunk_id,
            content=result1.content,
            score=fused_score,
            metadata={
                **result1.metadata,
                **result2.metadata,
                f"{source1}_score": result1.score,
                f"{source2}_score": result2.score,
                "fusion_method": self.method
            }
        )
        
        return fused_result