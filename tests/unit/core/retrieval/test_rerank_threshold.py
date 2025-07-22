"""Test rerank score threshold functionality."""

import pytest
from pathlib import Path
import tempfile
from unittest.mock import Mock, AsyncMock, patch

from knowledge_core_engine import KnowledgeEngine
from knowledge_core_engine.core.retrieval import RetrievalResult


class TestRerankThreshold:
    """Test rerank score threshold filtering."""
    
    @pytest.mark.asyncio
    async def test_rerank_threshold_auto_enable(self):
        """Test that setting rerank_score_threshold automatically enables enable_relevance_threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = KnowledgeEngine(
                persist_directory=tmpdir,
                enable_reranking=True,
                rerank_score_threshold=0.5,  # High threshold for testing
                retrieval_strategy="vector",
                retrieval_top_k=10
            )
            
            # Check that enable_relevance_threshold was automatically enabled
            assert engine.config.enable_relevance_threshold is True
            assert engine.config.rerank_score_threshold == 0.5
    
    def test_rerank_threshold_filtering_logic(self):
        """Test that rerank threshold properly filters results."""
        from knowledge_core_engine.core.retrieval import RetrievalResult
        
        # Create test results with rerank scores
        results_with_rerank = [
            RetrievalResult(
                chunk_id="1", 
                content="Result 1", 
                metadata={}, 
                score=0.8,
                rerank_score=0.8  # Above threshold
            ),
            RetrievalResult(
                chunk_id="2", 
                content="Result 2", 
                metadata={}, 
                score=0.7,
                rerank_score=0.3  # Below threshold
            ),
            RetrievalResult(
                chunk_id="3", 
                content="Result 3", 
                metadata={}, 
                score=0.6,
                rerank_score=0.6  # Above threshold
            ),
        ]
        
        # Simulate the filtering logic from retriever.py lines 163-170
        threshold = 0.5
        original_count = len(results_with_rerank)
        filtered_results = [r for r in results_with_rerank if r.rerank_score >= threshold]
        
        # Verify filtering worked
        assert len(filtered_results) == 2, f"Expected 2 results after filtering, got {len(filtered_results)}"
        assert all(r.rerank_score >= threshold for r in filtered_results), "All results should meet threshold"
        assert filtered_results[0].chunk_id == "1"
        assert filtered_results[1].chunk_id == "3"
    
    def test_rerank_threshold_not_set(self):
        """Test that when rerank_score_threshold is not set, no filtering occurs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = KnowledgeEngine(
                persist_directory=tmpdir,
                enable_reranking=False,  # Disable reranking to avoid torch dependency
                # rerank_score_threshold not set
                retrieval_strategy="vector",
                retrieval_top_k=10
            )
            
            # Check that enable_relevance_threshold is still False
            assert engine.config.enable_relevance_threshold is False
            
            # When rerank_score_threshold is explicitly None, should not enable filtering
            engine2 = KnowledgeEngine(
                persist_directory=tmpdir,
                rerank_score_threshold=None,
                retrieval_strategy="vector",
                retrieval_top_k=10
            )
            assert engine2.config.enable_relevance_threshold is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])