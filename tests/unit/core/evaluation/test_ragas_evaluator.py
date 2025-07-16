"""Tests for Ragas evaluator."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from knowledge_core_engine.core.evaluation import (
    TestCase,
    RagasEvaluator,
    RagasConfig,
    create_ragas_evaluator,
    RAGAS_AVAILABLE
)


@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="Ragas not installed")
class TestRagasEvaluator:
    """Test Ragas evaluator functionality."""
    
    @pytest.fixture
    def ragas_config(self):
        """Create test Ragas configuration."""
        return RagasConfig(
            llm_provider="deepseek",
            llm_api_key="test_key",
            embedding_provider="dashscope",
            embedding_api_key="test_key",
            metrics=["faithfulness", "answer_relevancy"]
        )
    
    @pytest.fixture
    def test_case(self):
        """Create test case."""
        return TestCase(
            question="What is RAG?",
            ground_truth="RAG is Retrieval-Augmented Generation",
            contexts=["RAG combines retrieval and generation"],
            generated_answer="RAG is a technique that combines retrieval and generation"
        )
    
    @pytest.mark.asyncio
    async def test_evaluator_initialization(self, ragas_config):
        """Test evaluator initialization."""
        with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.ChatOpenAI') as mock_llm:
            with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.DashScopeEmbeddings') as mock_emb:
                evaluator = RagasEvaluator(ragas_config)
                await evaluator.initialize()
                
                assert evaluator._initialized
                assert evaluator._llm is not None
                assert evaluator._embeddings is not None
                assert len(evaluator._metrics) == 2
    
    @pytest.mark.asyncio
    async def test_single_evaluation(self, ragas_config, test_case):
        """Test single test case evaluation."""
        with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.evaluate') as mock_evaluate:
            # Mock Ragas evaluate response
            mock_evaluate.return_value = {
                'faithfulness': 0.85,
                'answer_relevancy': 0.90
            }
            
            with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.ChatOpenAI'):
                with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.DashScopeEmbeddings'):
                    evaluator = RagasEvaluator(ragas_config)
                    await evaluator.initialize()
                    
                    result = await evaluator.evaluate_single(test_case)
                    
                    assert result.test_case_id == test_case.test_case_id
                    assert len(result.metrics) == 2
                    assert result.metrics[0].name == 'faithfulness'
                    assert result.metrics[0].score == 0.85
                    assert result.metrics[1].name == 'answer_relevancy'
                    assert result.metrics[1].score == 0.90
    
    @pytest.mark.asyncio
    async def test_batch_evaluation(self, ragas_config):
        """Test batch evaluation."""
        test_cases = [
            TestCase(
                question=f"Question {i}",
                ground_truth=f"Answer {i}",
                contexts=[f"Context {i}"],
                generated_answer=f"Generated {i}"
            )
            for i in range(3)
        ]
        
        with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.evaluate') as mock_evaluate:
            # Mock Ragas evaluate response
            import pandas as pd
            mock_evaluate.return_value = pd.DataFrame({
                'faithfulness': [0.8, 0.85, 0.9],
                'answer_relevancy': [0.85, 0.9, 0.95]
            })
            
            with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.ChatOpenAI'):
                with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.DashScopeEmbeddings'):
                    evaluator = RagasEvaluator(ragas_config)
                    evaluator.config.batch_size = 2  # Test batching
                    await evaluator.initialize()
                    
                    results = await evaluator.evaluate_batch(test_cases, show_progress=False)
                    
                    assert len(results) == 3
                    for i, result in enumerate(results):
                        assert result.test_case_id == test_cases[i].test_case_id
                        assert len(result.metrics) == 2
    
    def test_report_generation(self, ragas_config):
        """Test report generation."""
        from knowledge_core_engine.core.evaluation import EvaluationResult, MetricResult
        
        results = [
            EvaluationResult(
                test_case_id=f"tc_{i}",
                generated_answer=f"Answer {i}",
                metrics=[
                    MetricResult(name="faithfulness", score=0.8 + i * 0.05),
                    MetricResult(name="answer_relevancy", score=0.85 + i * 0.05)
                ]
            )
            for i in range(3)
        ]
        
        evaluator = RagasEvaluator(ragas_config)
        report = evaluator.generate_report(results)
        
        assert 'total_evaluations' in report
        assert report['total_evaluations'] == 3
        assert 'metrics' in report
        assert 'faithfulness' in report['metrics']
        assert 'answer_relevancy' in report['metrics']
        assert report['metrics']['faithfulness']['mean'] == 0.85
        assert report['metrics']['answer_relevancy']['mean'] == 0.90
    
    @pytest.mark.asyncio
    async def test_create_ragas_evaluator(self, ragas_config):
        """Test factory function."""
        with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.ChatOpenAI'):
            with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.DashScopeEmbeddings'):
                evaluator = await create_ragas_evaluator(ragas_config)
                
                assert isinstance(evaluator, RagasEvaluator)
                assert evaluator._initialized
    
    @pytest.mark.asyncio
    async def test_error_handling(self, ragas_config, test_case):
        """Test error handling in evaluation."""
        with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.evaluate') as mock_evaluate:
            # Mock evaluation failure
            mock_evaluate.side_effect = Exception("Evaluation failed")
            
            with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.ChatOpenAI'):
                with patch('knowledge_core_engine.core.evaluation.ragas_evaluator.DashScopeEmbeddings'):
                    evaluator = RagasEvaluator(ragas_config)
                    await evaluator.initialize()
                    
                    result = await evaluator.evaluate_single(test_case)
                    
                    assert result.test_case_id == test_case.test_case_id
                    assert len(result.metrics) == 0
                    assert 'error' in result.metadata
                    assert "Evaluation failed" in result.metadata['error']