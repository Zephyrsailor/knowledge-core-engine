"""Unit tests for the evaluation module."""

import pytest
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from knowledge_core_engine.core.evaluation.evaluator import (
    Evaluator,
    EvaluationConfig,
    EvaluationResult,
    MetricResult,
    TestCase,
    EvaluationMetrics
)


class TestEvaluationDataStructures:
    """Test evaluation data structures."""
    
    def test_test_case_creation(self):
        """Test creating a test case."""
        test_case = TestCase(
            question="什么是RAG技术？",
            ground_truth="RAG是检索增强生成技术，结合了信息检索和文本生成。",
            contexts=[
                "RAG (Retrieval-Augmented Generation) 是一种AI技术...",
                "检索增强生成通过先检索相关信息，再生成答案..."
            ],
            metadata={"category": "技术概念", "difficulty": "easy"}
        )
        
        assert test_case.question == "什么是RAG技术？"
        assert len(test_case.contexts) == 2
        assert test_case.metadata["category"] == "技术概念"
    
    def test_metric_result_creation(self):
        """Test creating metric results."""
        metric = MetricResult(
            name="faithfulness",
            score=0.85,
            confidence=0.92,
            details={
                "total_claims": 10,
                "supported_claims": 8,
                "unsupported_claims": 2
            }
        )
        
        assert metric.name == "faithfulness"
        assert metric.score == 0.85
        assert metric.details["supported_claims"] == 8
    
    def test_evaluation_result_aggregation(self):
        """Test evaluation result with multiple metrics."""
        metrics = [
            MetricResult("faithfulness", 0.85),
            MetricResult("answer_relevancy", 0.92),
            MetricResult("context_precision", 0.78)
        ]
        
        result = EvaluationResult(
            test_case_id="test_001",
            generated_answer="RAG是一种结合检索和生成的AI技术。",
            metrics=metrics,
            timestamp=datetime.now()
        )
        
        assert len(result.metrics) == 3
        assert result.get_metric("faithfulness").score == 0.85
        assert result.average_score() == pytest.approx(0.85, rel=0.01)


class TestEvaluationConfig:
    """Test evaluation configuration."""
    
    def test_default_config(self):
        """Test default evaluation config."""
        config = EvaluationConfig()
        
        assert config.metrics == ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        assert config.llm_provider == "deepseek"
        assert config.batch_size == 10
        assert config.use_cache is True
    
    def test_custom_config(self):
        """Test custom evaluation config."""
        config = EvaluationConfig(
            metrics=["faithfulness", "answer_relevancy"],
            llm_provider="qwen",
            llm_model="qwen-max",
            batch_size=20,
            use_cache=False
        )
        
        assert len(config.metrics) == 2
        assert config.llm_provider == "qwen"
        assert config.batch_size == 20
        assert config.use_cache is False
    
    def test_config_validation(self):
        """Test config validation."""
        # Invalid metric
        with pytest.raises(ValueError, match="Unknown metric"):
            EvaluationConfig(metrics=["invalid_metric"])
        
        # Invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            EvaluationConfig(batch_size=0)


class TestEvaluator:
    """Test the main evaluator class."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return EvaluationConfig(
            metrics=["faithfulness", "answer_relevancy"],
            llm_provider="mock"
        )
    
    @pytest.fixture
    def evaluator(self, config):
        """Create evaluator instance."""
        return Evaluator(config)
    
    @pytest.fixture
    def sample_test_cases(self):
        """Create sample test cases."""
        return [
            TestCase(
                question="什么是RAG？",
                ground_truth="RAG是检索增强生成技术。",
                contexts=["RAG技术结合了检索和生成..."],
                generated_answer="RAG是一种AI技术，用于改善生成质量。"
            ),
            TestCase(
                question="RAG的优势是什么？",
                ground_truth="RAG的优势包括准确性高、可追溯性强。",
                contexts=["RAG技术的主要优势..."],
                generated_answer="RAG可以提供更准确的答案。"
            )
        ]
    
    def test_evaluator_initialization(self, evaluator, config):
        """Test evaluator initialization."""
        assert evaluator.config == config
        assert evaluator._metrics_registry is not None
        assert "faithfulness" in evaluator._metrics_registry
    
    @pytest.mark.asyncio
    async def test_evaluate_single_test_case(self, evaluator, sample_test_cases):
        """Test evaluating a single test case."""
        test_case = sample_test_cases[0]
        
        # Mock the metric evaluation
        async def mock_faithfulness(answer, contexts, **kwargs):
            return MetricResult("faithfulness", 0.85, details={"mock": True})
        
        async def mock_relevancy(answer, question, **kwargs):
            return MetricResult("answer_relevancy", 0.90, details={"mock": True})
        
        evaluator._metrics_registry["faithfulness"] = mock_faithfulness
        evaluator._metrics_registry["answer_relevancy"] = mock_relevancy
        
        result = await evaluator.evaluate_single(test_case)
        
        assert result.test_case_id is not None
        assert result.generated_answer == test_case.generated_answer
        assert len(result.metrics) == 2
        assert result.get_metric("faithfulness").score == 0.85
        assert result.get_metric("answer_relevancy").score == 0.90
    
    @pytest.mark.asyncio
    async def test_batch_evaluation(self, evaluator, sample_test_cases):
        """Test batch evaluation."""
        # Mock metrics
        async def mock_metric(answer, **kwargs):
            return MetricResult("mock_metric", 0.8)
        
        evaluator._metrics_registry = {
            "faithfulness": mock_metric,
            "answer_relevancy": mock_metric
        }
        
        results = await evaluator.evaluate_batch(sample_test_cases)
        
        assert len(results) == 2
        for result in results:
            assert len(result.metrics) == 2
            assert all(m.score == 0.8 for m in result.metrics)
    
    @pytest.mark.asyncio
    async def test_evaluation_with_custom_metrics(self, config):
        """Test evaluation with custom metrics."""
        # Define custom metric
        async def custom_length_metric(answer: str, **kwargs) -> MetricResult:
            score = min(len(answer) / 100, 1.0)  # Normalize by 100 chars
            return MetricResult(
                name="answer_length",
                score=score,
                details={"length": len(answer)}
            )
        
        # Create evaluator with custom metric
        evaluator = Evaluator(config)
        evaluator.register_metric("answer_length", custom_length_metric)
        
        # Update config to use custom metric
        evaluator.config.metrics = ["answer_length"]
        
        test_case = TestCase(
            question="Test question",
            ground_truth="Test answer",
            contexts=["Test context"],
            generated_answer="This is a test answer with some length."
        )
        
        result = await evaluator.evaluate_single(test_case)
        
        assert result.get_metric("answer_length") is not None
        assert result.get_metric("answer_length").details["length"] == len(test_case.generated_answer)
    
    def test_evaluation_summary(self, evaluator):
        """Test generating evaluation summary."""
        results = [
            EvaluationResult(
                test_case_id="1",
                generated_answer="Answer 1",
                metrics=[
                    MetricResult("faithfulness", 0.8),
                    MetricResult("answer_relevancy", 0.9)
                ]
            ),
            EvaluationResult(
                test_case_id="2", 
                generated_answer="Answer 2",
                metrics=[
                    MetricResult("faithfulness", 0.7),
                    MetricResult("answer_relevancy", 0.85)
                ]
            )
        ]
        
        summary = evaluator.generate_summary(results)
        
        assert summary["total_test_cases"] == 2
        assert summary["average_scores"]["faithfulness"] == 0.75
        assert summary["average_scores"]["answer_relevancy"] == 0.875
        assert summary["overall_score"] == pytest.approx(0.8125, rel=0.01)


class TestEvaluationMetrics:
    """Test individual evaluation metrics."""
    
    @pytest.mark.asyncio
    async def test_faithfulness_metric(self):
        """Test faithfulness metric calculation."""
        from knowledge_core_engine.core.evaluation.metrics import FaithfulnessMetric
        
        metric = FaithfulnessMetric()
        
        answer = "RAG技术结合了检索和生成，可以提高准确性。"
        contexts = [
            "RAG是检索增强生成技术，它先检索相关信息再生成答案。",
            "RAG的优势包括提高生成准确性和可追溯性。"
        ]
        
        result = await metric.calculate(answer=answer, contexts=contexts)
        
        assert result.name == "faithfulness"
        assert 0 <= result.score <= 1
        assert "claims" in result.details
    
    @pytest.mark.asyncio
    async def test_answer_relevancy_metric(self):
        """Test answer relevancy metric."""
        from knowledge_core_engine.core.evaluation.metrics import AnswerRelevancyMetric
        
        metric = AnswerRelevancyMetric()
        
        question = "什么是RAG技术？"
        answer = "RAG是检索增强生成技术，用于改善AI生成质量。"
        
        result = await metric.calculate(question=question, answer=answer)
        
        assert result.name == "answer_relevancy"
        assert 0 <= result.score <= 1
    
    @pytest.mark.asyncio
    async def test_context_precision_metric(self):
        """Test context precision metric."""
        from knowledge_core_engine.core.evaluation.metrics import ContextPrecisionMetric
        
        metric = ContextPrecisionMetric()
        
        question = "RAG的优势？"
        contexts = [
            "RAG技术的优势包括准确性高。",  # Relevant
            "Python是一种编程语言。",  # Irrelevant
            "RAG可以追溯信息来源。"  # Relevant
        ]
        ground_truth = "RAG的优势是准确性高和可追溯性。"
        
        result = await metric.calculate(
            question=question,
            contexts=contexts,
            ground_truth=ground_truth
        )
        
        assert result.name == "context_precision"
        assert result.score == pytest.approx(0.67, rel=0.1)  # 2/3 relevant


class TestGoldenDataset:
    """Test golden dataset management."""
    
    def test_load_golden_dataset(self):
        """Test loading golden dataset from file."""
        from knowledge_core_engine.core.evaluation.golden_dataset import GoldenDataset
        
        dataset = GoldenDataset()
        
        # Test adding test cases
        dataset.add_test_case(
            question="什么是RAG？",
            ground_truth="RAG是检索增强生成技术。",
            contexts=["RAG技术说明..."],
            metadata={"category": "definition"}
        )
        
        assert len(dataset) == 1
        assert dataset[0].question == "什么是RAG？"
    
    def test_dataset_filtering(self):
        """Test filtering dataset by metadata."""
        from knowledge_core_engine.core.evaluation.golden_dataset import GoldenDataset
        
        dataset = GoldenDataset()
        
        # Add test cases with different categories
        dataset.add_test_case("Q1", "A1", ["C1"], {"category": "easy"})
        dataset.add_test_case("Q2", "A2", ["C2"], {"category": "hard"})
        dataset.add_test_case("Q3", "A3", ["C3"], {"category": "easy"})
        
        # Filter by category
        easy_cases = dataset.filter_by(category="easy")
        
        assert len(easy_cases) == 2
        assert all(tc.metadata["category"] == "easy" for tc in easy_cases)
    
    def test_dataset_io(self, tmp_path):
        """Test saving and loading dataset."""
        from knowledge_core_engine.core.evaluation.golden_dataset import GoldenDataset
        
        dataset = GoldenDataset()
        dataset.add_test_case("Q1", "A1", ["C1"])
        dataset.add_test_case("Q2", "A2", ["C2"])
        
        # Save dataset
        file_path = tmp_path / "test_dataset.json"
        dataset.save(file_path)
        
        # Load dataset
        loaded = GoldenDataset.load(file_path)
        
        assert len(loaded) == 2
        assert loaded[0].question == "Q1"
        assert loaded[1].question == "Q2"