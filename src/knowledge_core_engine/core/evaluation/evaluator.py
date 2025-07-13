"""Core evaluator for RAG system performance assessment."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import statistics

from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class EvaluationMetrics(str, Enum):
    """Available evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    ANSWER_SIMILARITY = "answer_similarity"
    ANSWER_CORRECTNESS = "answer_correctness"


@dataclass
class TestCase:
    """A single test case for evaluation."""
    question: str
    ground_truth: str
    contexts: List[str]
    generated_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    test_case_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.test_case_id:
            import hashlib
            content = f"{self.question}:{self.ground_truth}"
            self.test_case_id = hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""
    name: str
    score: float
    confidence: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate score range."""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")


@dataclass
class EvaluationResult:
    """Result from evaluating a single test case."""
    test_case_id: str
    generated_answer: str
    metrics: List[MetricResult]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_metric(self, name: str) -> Optional[MetricResult]:
        """Get specific metric result by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None
    
    def average_score(self) -> float:
        """Calculate average score across all metrics."""
        if not self.metrics:
            return 0.0
        return statistics.mean(m.score for m in self.metrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_case_id": self.test_case_id,
            "generated_answer": self.generated_answer,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "confidence": m.confidence,
                    "details": m.details
                } for m in self.metrics
            ],
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "average_score": self.average_score()
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Metrics to evaluate
    metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy", 
        "context_precision",
        "context_recall"
    ])
    
    # LLM configuration for evaluation
    llm_provider: str = "deepseek"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    temperature: float = 0.0  # Use deterministic evaluation
    
    # Processing configuration
    batch_size: int = 10
    max_concurrent_evaluations: int = 5
    timeout_seconds: int = 60
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    
    # Output configuration
    output_format: str = "json"  # json, csv, markdown
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate metrics
        valid_metrics = {m.value for m in EvaluationMetrics}
        for metric in self.metrics:
            if metric not in valid_metrics and not metric.startswith("custom_"):
                raise ValueError(f"Unknown metric: {metric}. Valid metrics: {valid_metrics}")
        
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Set default cache directory
        if self.use_cache and not self.cache_dir:
            self.cache_dir = Path.home() / ".cache" / "knowledge_core_engine" / "evaluation"


class Evaluator:
    """Main evaluator for RAG system performance."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self._metrics_registry: Dict[str, Callable] = {}
        self._cache: Dict[str, Any] = {}
        self._initialized = False
        
        # Register built-in metrics
        self._register_builtin_metrics()
        
        logger.info(f"Evaluator initialized with metrics: {self.config.metrics}")
    
    def _register_builtin_metrics(self):
        """Register built-in evaluation metrics."""
        # These will be implemented in the metrics module
        from . import metrics
        
        self._metrics_registry = {
            "faithfulness": self._create_faithfulness_metric,
            "answer_relevancy": self._create_answer_relevancy_metric,
            "context_precision": self._create_context_precision_metric,
            "context_recall": self._create_context_recall_metric,
        }
    
    async def _create_faithfulness_metric(self, answer: str, contexts: List[str], **kwargs) -> MetricResult:
        """Create faithfulness metric (placeholder)."""
        # This will be implemented with actual metric
        score = 0.85  # Mock score
        return MetricResult(
            name="faithfulness",
            score=score,
            details={"method": "mock"}
        )
    
    async def _create_answer_relevancy_metric(self, answer: str, question: str, **kwargs) -> MetricResult:
        """Create answer relevancy metric (placeholder)."""
        score = 0.90  # Mock score
        return MetricResult(
            name="answer_relevancy",
            score=score,
            details={"method": "mock"}
        )
    
    async def _create_context_precision_metric(self, contexts: List[str], question: str, 
                                                ground_truth: str, **kwargs) -> MetricResult:
        """Create context precision metric (placeholder)."""
        score = 0.80  # Mock score
        return MetricResult(
            name="context_precision",
            score=score,
            details={"method": "mock"}
        )
    
    async def _create_context_recall_metric(self, contexts: List[str], ground_truth: str, **kwargs) -> MetricResult:
        """Create context recall metric (placeholder)."""
        score = 0.75  # Mock score
        return MetricResult(
            name="context_recall",
            score=score,
            details={"method": "mock"}
        )
    
    def register_metric(self, name: str, metric_func: Callable) -> None:
        """Register a custom metric function.
        
        Args:
            name: Name of the metric
            metric_func: Async function that calculates the metric
        """
        self._metrics_registry[name] = metric_func
        logger.info(f"Registered custom metric: {name}")
    
    async def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case.
        
        Args:
            test_case: Test case to evaluate
            
        Returns:
            Evaluation result with all metrics
        """
        logger.debug(f"Evaluating test case: {test_case.test_case_id}")
        
        # Ensure we have a generated answer
        if not test_case.generated_answer:
            raise ValueError("Test case must have a generated_answer for evaluation")
        
        # Calculate each metric
        metric_results = []
        
        for metric_name in self.config.metrics:
            if metric_name not in self._metrics_registry:
                logger.warning(f"Metric {metric_name} not found in registry, skipping")
                continue
            
            try:
                # Get metric function
                metric_func = self._metrics_registry[metric_name]
                
                # Calculate metric
                result = await metric_func(
                    answer=test_case.generated_answer,
                    question=test_case.question,
                    contexts=test_case.contexts,
                    ground_truth=test_case.ground_truth,
                    metadata=test_case.metadata
                )
                
                metric_results.append(result)
                
            except Exception as e:
                logger.error(f"Error calculating metric {metric_name}: {e}")
                # Add failed metric with score 0
                metric_results.append(MetricResult(
                    name=metric_name,
                    score=0.0,
                    details={"error": str(e)}
                ))
        
        return EvaluationResult(
            test_case_id=test_case.test_case_id,
            generated_answer=test_case.generated_answer,
            metrics=metric_results,
            metadata=test_case.metadata
        )
    
    async def evaluate_batch(self, test_cases: List[TestCase], 
                           show_progress: bool = True) -> List[EvaluationResult]:
        """Evaluate multiple test cases.
        
        Args:
            test_cases: List of test cases to evaluate
            show_progress: Whether to show progress
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Evaluating batch of {len(test_cases)} test cases")
        
        results = []
        
        # Process in batches
        for i in range(0, len(test_cases), self.config.batch_size):
            batch = test_cases[i:i + self.config.batch_size]
            
            # Evaluate batch concurrently
            batch_tasks = [self.evaluate_single(tc) for tc in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error evaluating test case {i+j}: {result}")
                    # Create empty result for failed evaluation
                    results.append(EvaluationResult(
                        test_case_id=batch[j].test_case_id,
                        generated_answer=batch[j].generated_answer or "",
                        metrics=[],
                        metadata={"error": str(result)}
                    ))
                else:
                    results.append(result)
            
            if show_progress:
                progress = (i + len(batch)) / len(test_cases) * 100
                logger.info(f"Progress: {progress:.1f}%")
        
        return results
    
    def generate_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate summary statistics from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Summary dictionary with statistics
        """
        if not results:
            return {
                "total_test_cases": 0,
                "average_scores": {},
                "overall_score": 0.0
            }
        
        # Calculate average for each metric
        metric_scores: Dict[str, List[float]] = {}
        
        for result in results:
            for metric in result.metrics:
                if metric.name not in metric_scores:
                    metric_scores[metric.name] = []
                metric_scores[metric.name].append(metric.score)
        
        # Calculate averages
        average_scores = {
            metric: statistics.mean(scores)
            for metric, scores in metric_scores.items()
        }
        
        # Calculate overall score
        all_scores = []
        for scores in metric_scores.values():
            all_scores.extend(scores)
        
        overall_score = statistics.mean(all_scores) if all_scores else 0.0
        
        # Calculate additional statistics
        summary = {
            "total_test_cases": len(results),
            "average_scores": average_scores,
            "overall_score": overall_score,
            "score_distribution": {
                metric: {
                    "min": min(scores),
                    "max": max(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0.0
                }
                for metric, scores in metric_scores.items()
            },
            "failed_evaluations": sum(1 for r in results if not r.metrics),
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    async def save_results(self, results: List[EvaluationResult], 
                          output_path: Path) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results to save
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == "json":
            data = {
                "results": [r.to_dict() for r in results],
                "summary": self.generate_summary(results),
                "config": {
                    "metrics": self.config.metrics,
                    "llm_provider": self.config.llm_provider,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        elif self.config.output_format == "csv":
            # TODO: Implement CSV output
            raise NotImplementedError("CSV output not yet implemented")
        
        elif self.config.output_format == "markdown":
            # TODO: Implement Markdown report
            raise NotImplementedError("Markdown output not yet implemented")
        
        else:
            raise ValueError(f"Unknown output format: {self.config.output_format}")
        
        logger.info(f"Results saved to {output_path}")