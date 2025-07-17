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

# Ragas integration (optional)
try:
    from .ragas_evaluator import (
        RagasEvaluator,
        RagasConfig,
        create_ragas_evaluator
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    RagasEvaluator = None
    RagasConfig = None
    create_ragas_evaluator = None

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
    "GoldenDataset",
    
    # Ragas integration
    "RagasEvaluator",
    "RagasConfig",
    "create_ragas_evaluator",
    "RAGAS_AVAILABLE"
]