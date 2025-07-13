"""Unit tests for hybrid retrieval strategy."""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.retrieval.hybrid_strategy import (
    HybridRetriever, ScoreFusion, ReciprocalRankFusion
)
from knowledge_core_engine.core.retrieval.retriever import RetrievalResult
from knowledge_core_engine.core.embedding.vector_store import QueryResult


class TestHybridRetriever:
    """Test hybrid retrieval strategy."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGConfig(
            retrieval_strategy="hybrid",
            extra_params={
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "fusion_method": "weighted",
                "normalization": "min_max"
            }
        )
    
    @pytest.fixture
    def hybrid_retriever(self, config):
        """Create HybridRetriever instance."""
        return HybridRetriever(config)
    
    @pytest.fixture
    def vector_results(self):
        """Mock vector search results."""
        return [
            QueryResult(
                id="doc_1",
                score=0.95,
                text="深度学习是机器学习的一个分支",
                metadata={"source": "vector", "doc_type": "tutorial"}
            ),
            QueryResult(
                id="doc_2",
                score=0.88,
                text="神经网络的基本原理",
                metadata={"source": "vector", "doc_type": "theory"}
            ),
            QueryResult(
                id="doc_3",
                score=0.82,
                text="卷积神经网络在图像识别中的应用",
                metadata={"source": "vector", "doc_type": "application"}
            ),
            QueryResult(
                id="doc_5",
                score=0.75,
                text="循环神经网络处理序列数据",
                metadata={"source": "vector", "doc_type": "theory"}
            )
        ]
    
    @pytest.fixture
    def bm25_results(self):
        """Mock BM25 search results."""
        return [
            {
                "id": "doc_2",  # Overlap with vector
                "score": 12.5,
                "text": "神经网络的基本原理",
                "metadata": {"source": "bm25", "doc_type": "theory"}
            },
            {
                "id": "doc_4",
                "score": 10.8,
                "text": "深度学习框架比较：TensorFlow vs PyTorch",
                "metadata": {"source": "bm25", "doc_type": "comparison"}
            },
            {
                "id": "doc_1",  # Overlap with vector
                "score": 9.2,
                "text": "深度学习是机器学习的一个分支",
                "metadata": {"source": "bm25", "doc_type": "tutorial"}
            },
            {
                "id": "doc_6",
                "score": 7.5,
                "text": "强化学习在游戏AI中的应用",
                "metadata": {"source": "bm25", "doc_type": "application"}
            }
        ]
    
    @pytest.mark.asyncio
    async def test_weighted_fusion(self, hybrid_retriever, vector_results, bm25_results):
        """Test weighted score fusion."""
        # Normalize scores first
        vector_normalized = hybrid_retriever._normalize_scores(
            vector_results, method="min_max"
        )
        bm25_normalized = hybrid_retriever._normalize_scores(
            bm25_results, method="min_max"
        )
        
        # Apply weighted fusion
        fused_results = hybrid_retriever._weighted_fusion(
            vector_normalized,
            bm25_normalized,
            vector_weight=0.7,
            bm25_weight=0.3
        )
        
        # Check results
        assert len(fused_results) > 0
        
        # Find doc_1 (appears in both)
        doc_1_result = next(r for r in fused_results if r.chunk_id == "doc_1")
        assert doc_1_result is not None
        
        # Score should be weighted combination
        # doc_1: vector=0.95 (normalized to 1.0), bm25=9.2 (normalized)
        assert 0 <= doc_1_result.score <= 1.0
        
        # Check metadata preservation
        assert "vector_score" in doc_1_result.metadata
        assert "bm25_score" in doc_1_result.metadata
        assert "fusion_method" in doc_1_result.metadata
    
    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion(self, hybrid_retriever, vector_results, bm25_results):
        """Test Reciprocal Rank Fusion (RRF)."""
        k = 60  # Standard RRF parameter
        
        rrf = ReciprocalRankFusion(k=k)
        fused_results = rrf.fuse(vector_results, bm25_results)
        
        # Check results
        assert len(fused_results) > 0
        
        # RRF scores should be between 0 and 1/k * 2 (for appearing in both lists)
        for result in fused_results:
            assert 0 < result.score <= 2.0 / k
        
        # Results should be sorted by RRF score
        for i in range(1, len(fused_results)):
            assert fused_results[i-1].score >= fused_results[i].score
    
    def test_score_normalization_min_max(self, hybrid_retriever):
        """Test min-max normalization."""
        results = [
            {"id": "1", "score": 10.0},
            {"id": "2", "score": 5.0},
            {"id": "3", "score": 15.0},
            {"id": "4", "score": 0.0}
        ]
        
        normalized = hybrid_retriever._normalize_scores(results, method="min_max")
        
        # Check normalization
        scores = [r["score"] for r in normalized]
        assert min(scores) == 0.0
        assert max(scores) == 1.0
        
        # Check relative ordering preserved
        assert normalized[2]["score"] == 1.0  # Was 15.0 (max)
        assert normalized[3]["score"] == 0.0  # Was 0.0 (min)
    
    def test_score_normalization_z_score(self, hybrid_retriever):
        """Test z-score normalization."""
        results = [
            {"id": "1", "score": 10.0},
            {"id": "2", "score": 20.0},
            {"id": "3", "score": 15.0},
            {"id": "4", "score": 25.0},
            {"id": "5", "score": 30.0}
        ]
        
        normalized = hybrid_retriever._normalize_scores(results, method="z_score")
        
        # Check z-score properties
        scores = [r["score"] for r in normalized]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Mean should be close to 0, std close to 1
        assert abs(mean_score) < 0.01
        assert abs(std_score - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_full_pipeline(self, hybrid_retriever):
        """Test full hybrid retrieval pipeline."""
        query = "深度学习基础"
        
        with patch.object(hybrid_retriever, '_vector_search', new_callable=AsyncMock) as mock_vector, \
             patch.object(hybrid_retriever, '_bm25_search', new_callable=AsyncMock) as mock_bm25:
            
            # Mock search results
            mock_vector.return_value = [
                QueryResult("vec_1", 0.9, "Vector result 1", {}),
                QueryResult("vec_2", 0.8, "Vector result 2", {}),
                QueryResult("common_1", 0.75, "Common result", {})
            ]
            
            mock_bm25.return_value = [
                {"id": "bm25_1", "score": 15.0, "text": "BM25 result 1", "metadata": {}},
                {"id": "common_1", "score": 12.0, "text": "Common result", "metadata": {}},
                {"id": "bm25_2", "score": 10.0, "text": "BM25 result 2", "metadata": {}}
            ]
            
            results = await hybrid_retriever.retrieve(query, top_k=5)
            
            assert len(results) <= 5
            assert all(isinstance(r, RetrievalResult) for r in results)
            
            # Check deduplication
            ids = [r.chunk_id for r in results]
            assert len(ids) == len(set(ids))  # No duplicates
            
            # Check that common_1 has combined scores
            common_result = next((r for r in results if r.chunk_id == "common_1"), None)
            if common_result:
                assert "vector_score" in common_result.metadata
                assert "bm25_score" in common_result.metadata
    
    @pytest.mark.asyncio
    async def test_hybrid_with_filters(self, hybrid_retriever):
        """Test hybrid retrieval with metadata filters."""
        query = "机器学习教程"
        filters = {"doc_type": "tutorial"}
        
        with patch.object(hybrid_retriever, '_vector_search', new_callable=AsyncMock) as mock_vector, \
             patch.object(hybrid_retriever, '_bm25_search', new_callable=AsyncMock) as mock_bm25:
            
            # Only return filtered results
            mock_vector.return_value = [
                QueryResult("doc_1", 0.9, "Tutorial 1", {"doc_type": "tutorial"})
            ]
            
            mock_bm25.return_value = [
                {"id": "doc_2", "score": 10.0, "text": "Tutorial 2", 
                 "metadata": {"doc_type": "tutorial"}}
            ]
            
            results = await hybrid_retriever.retrieve(query, filters=filters)
            
            # All results should match filter
            assert all(r.metadata.get("doc_type") == "tutorial" for r in results)
    
    @pytest.mark.asyncio
    async def test_hybrid_empty_results_handling(self, hybrid_retriever):
        """Test handling when one search returns empty results."""
        query = "very specific query"
        
        with patch.object(hybrid_retriever, '_vector_search', new_callable=AsyncMock) as mock_vector, \
             patch.object(hybrid_retriever, '_bm25_search', new_callable=AsyncMock) as mock_bm25:
            
            # Vector returns results, BM25 returns empty
            mock_vector.return_value = [
                QueryResult("vec_1", 0.9, "Vector only result", {})
            ]
            mock_bm25.return_value = []
            
            results = await hybrid_retriever.retrieve(query)
            
            # Should still return vector results
            assert len(results) == 1
            assert results[0].chunk_id == "vec_1"
            
            # Now test opposite case
            mock_vector.return_value = []
            mock_bm25.return_value = [
                {"id": "bm25_1", "score": 10.0, "text": "BM25 only", "metadata": {}}
            ]
            
            results = await hybrid_retriever.retrieve(query)
            
            # Should still return BM25 results
            assert len(results) == 1
            assert results[0].chunk_id == "bm25_1"
    
    def test_score_fusion_strategies(self):
        """Test different score fusion strategies."""
        vector_result = RetrievalResult("doc_1", "Content", 0.9, {"source": "vector"})
        bm25_result = RetrievalResult("doc_1", "Content", 0.7, {"source": "bm25"})
        
        # Test MAX fusion
        fusion = ScoreFusion(method="max")
        fused = fusion.fuse_scores(vector_result, bm25_result)
        assert fused.score == 0.9
        
        # Test MIN fusion
        fusion = ScoreFusion(method="min")
        fused = fusion.fuse_scores(vector_result, bm25_result)
        assert fused.score == 0.7
        
        # Test MEAN fusion
        fusion = ScoreFusion(method="mean")
        fused = fusion.fuse_scores(vector_result, bm25_result)
        assert fused.score == 0.8
        
        # Test WEIGHTED fusion
        fusion = ScoreFusion(method="weighted", weights={"vector": 0.6, "bm25": 0.4})
        fused = fusion.fuse_scores(vector_result, bm25_result)
        assert abs(fused.score - (0.9 * 0.6 + 0.7 * 0.4)) < 0.001


class TestReciprocalRankFusion:
    """Test RRF implementation."""
    
    def test_rrf_basic(self):
        """Test basic RRF calculation."""
        rrf = ReciprocalRankFusion(k=60)
        
        # Create ranked lists
        list1 = [
            {"id": "A", "score": 0.9},
            {"id": "B", "score": 0.8},
            {"id": "C", "score": 0.7}
        ]
        
        list2 = [
            {"id": "B", "score": 12.0},
            {"id": "D", "score": 10.0},
            {"id": "A", "score": 8.0}
        ]
        
        fused = rrf.fuse(list1, list2)
        
        # Check RRF scores
        # A: rank 1 in list1, rank 3 in list2
        # RRF(A) = 1/(60+1) + 1/(60+3) ≈ 0.0164 + 0.0159
        a_result = next(r for r in fused if r.chunk_id == "A")
        expected_a = 1/(60+1) + 1/(60+3)
        assert abs(a_result.score - expected_a) < 0.0001
        
        # B should have highest score (rank 2 in list1, rank 1 in list2)
        b_result = next(r for r in fused if r.chunk_id == "B")
        assert b_result.score > a_result.score
    
    def test_rrf_single_list_appearance(self):
        """Test RRF when document appears in only one list."""
        rrf = ReciprocalRankFusion(k=60)
        
        list1 = [{"id": "A", "score": 0.9}]
        list2 = [{"id": "B", "score": 0.8}]
        
        fused = rrf.fuse(list1, list2)
        
        # Both should have same RRF score (rank 1 in their respective lists)
        assert len(fused) == 2
        assert abs(fused[0].score - fused[1].score) < 0.0001
    
    def test_rrf_with_many_lists(self):
        """Test RRF with more than 2 lists."""
        rrf = ReciprocalRankFusion(k=60)
        
        lists = [
            [{"id": "A", "score": 0.9}, {"id": "B", "score": 0.8}],
            [{"id": "B", "score": 0.85}, {"id": "C", "score": 0.75}],
            [{"id": "A", "score": 0.88}, {"id": "C", "score": 0.78}]
        ]
        
        fused = rrf.fuse_multiple(lists)
        
        # A appears in lists 0 and 2 at rank 1
        # B appears in lists 0 and 1 at ranks 2 and 1
        # C appears in lists 1 and 2 at rank 2
        
        # Check all documents are present
        ids = [r.chunk_id for r in fused]
        assert set(ids) == {"A", "B", "C"}


class TestAdvancedHybridStrategies:
    """Test advanced hybrid retrieval strategies."""
    
    @pytest.mark.asyncio
    async def test_cascade_retrieval(self):
        """Test cascade retrieval strategy."""
        # First stage: Fast BM25
        # Second stage: Expensive vector search on BM25 results
        # This tests efficiency optimization
        pass
    
    @pytest.mark.asyncio
    async def test_diversified_retrieval(self):
        """Test result diversification."""
        # Ensure results cover different aspects/topics
        # Implement MMR (Maximal Marginal Relevance)
        pass
    
    @pytest.mark.asyncio 
    async def test_adaptive_weighting(self):
        """Test adaptive weight adjustment based on query type."""
        # Short queries: higher BM25 weight
        # Long queries: higher vector weight
        # Question queries: balanced weights
        pass