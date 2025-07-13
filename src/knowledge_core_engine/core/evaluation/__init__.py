"""Evaluation module for assessing RAG system performance."""

from .evaluator import (
    Evaluator,
    EvaluationConfig,
    EvaluationResult,
    MetricResult,
    TestCase,
    EvaluationMetrics
)
from .metrics import (
    BaseMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric
)
from .golden_dataset import GoldenDataset

__all__ = [
    # Core evaluator
    "Evaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "MetricResult",
    "TestCase",
    "EvaluationMetrics",
    
    # Metrics
    "BaseMetric",
    "FaithfulnessMetric",
    "AnswerRelevancyMetric", 
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    
    # Dataset management
    "GoldenDataset"
]