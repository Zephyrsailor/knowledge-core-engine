"""Ragas-based evaluation implementation for RAG system."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("Ragas not installed. Install with: pip install ragas")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings

from ..config import RAGConfig
from .evaluator import TestCase, EvaluationResult, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class RagasConfig:
    """Configuration for Ragas evaluation."""
    # LLM configuration
    llm_provider: str = "qwen"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_api_base: Optional[str] = None
    
    # Embedding configuration
    embedding_provider: str = "dashscope"
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None
    
    # Metrics to evaluate
    metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ])
    
    # Processing
    batch_size: int = 10
    timeout: int = 60
    
    def __post_init__(self):
        """Set default models if not specified."""
        if not self.llm_model:
            if self.llm_provider == "deepseek":
                self.llm_model = "deepseek-chat"
            elif self.llm_provider == "qwen":
                self.llm_model = "qwen-turbo"  # 使用更快的模型
            elif self.llm_provider == "openai":
                self.llm_model = "gpt-4-turbo-preview"
        
        if not self.embedding_model:
            if self.embedding_provider == "dashscope":
                self.embedding_model = "text-embedding-v3"
            elif self.embedding_provider == "openai":
                self.embedding_model = "text-embedding-3-small"


class RagasEvaluator:
    """Evaluator using Ragas framework."""
    
    def __init__(self, config: Optional[RagasConfig] = None):
        """Initialize Ragas evaluator.
        
        Args:
            config: Ragas configuration
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Ragas is not installed. Install with: pip install ragas")
        
        self.config = config or RagasConfig()
        self._llm = None
        self._embeddings = None
        self._metrics = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize LLM and embedding models."""
        if self._initialized:
            return
        
        # Initialize LLM
        if self.config.llm_provider in ["deepseek", "openai"]:
            # DeepSeek uses OpenAI-compatible API
            api_base = self.config.llm_api_base
            if self.config.llm_provider == "deepseek" and not api_base:
                api_base = "https://api.deepseek.com/v1"
            
            self._llm = ChatOpenAI(
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                base_url=api_base,
                temperature=0
            )
        elif self.config.llm_provider == "qwen":
            self._llm = Tongyi(
                model_name=self.config.llm_model,
                dashscope_api_key=self.config.llm_api_key,
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
        
        # Initialize embeddings
        if self.config.embedding_provider == "dashscope":
            self._embeddings = DashScopeEmbeddings(
                model=self.config.embedding_model,
                dashscope_api_key=self.config.embedding_api_key
            )
        elif self.config.embedding_provider == "openai":
            self._embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=self.config.embedding_api_key
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.embedding_provider}")
        
        # Wrap for Ragas
        self._llm_wrapper = LangchainLLMWrapper(self._llm)
        self._embeddings_wrapper = LangchainEmbeddingsWrapper(self._embeddings)
        
        # Initialize metrics
        self._metrics = self._get_metrics()
        
        # Configure metrics to use our LLM and embeddings
        for metric in self._metrics:
            if hasattr(metric, 'llm'):
                metric.llm = self._llm_wrapper
            if hasattr(metric, 'embeddings'):
                metric.embeddings = self._embeddings_wrapper
        
        self._initialized = True
        logger.info(f"Ragas evaluator initialized with {self.config.llm_provider} LLM and {self.config.embedding_provider} embeddings")
    
    def _get_metrics(self) -> List[Any]:
        """Get Ragas metric objects based on configuration."""
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness
        }
        
        metrics = []
        for metric_name in self.config.metrics:
            if metric_name in metric_map:
                metrics.append(metric_map[metric_name])
            else:
                logger.warning(f"Unknown metric: {metric_name}")
        
        return metrics
    
    async def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case.
        
        Args:
            test_case: Test case to evaluate
            
        Returns:
            Evaluation result with metrics
        """
        await self.initialize()
        
        # Prepare data for Ragas
        data = {
            "question": [test_case.question],
            "answer": [test_case.generated_answer or ""],
            "contexts": [test_case.contexts],
            "ground_truth": [test_case.ground_truth]
        }
        
        dataset = pd.DataFrame(data)
        
        # Run evaluation
        try:
            result = evaluate(
                dataset=dataset,
                metrics=self._metrics,
                llm=self._llm_wrapper,
                embeddings=self._embeddings_wrapper,
                raise_exceptions=False
            )
            
            # Convert to our format
            metric_results = []
            for metric_name, score in result.items():
                if metric_name != "question":  # Skip non-metric columns
                    metric_results.append(MetricResult(
                        name=metric_name,
                        score=float(score) if not pd.isna(score) else 0.0,
                        confidence=0.9,
                        details={"ragas": True}
                    ))
            
            return EvaluationResult(
                test_case_id=test_case.test_case_id,
                generated_answer=test_case.generated_answer,
                metrics=metric_results,
                metadata={"evaluator": "ragas"}
            )
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            # Return empty result on failure
            return EvaluationResult(
                test_case_id=test_case.test_case_id,
                generated_answer=test_case.generated_answer,
                metrics=[],
                metadata={"error": str(e), "evaluator": "ragas"}
            )
    
    async def evaluate_batch(self, test_cases: List[TestCase], 
                           show_progress: bool = True) -> List[EvaluationResult]:
        """Evaluate multiple test cases.
        
        Args:
            test_cases: List of test cases
            show_progress: Whether to show progress
            
        Returns:
            List of evaluation results
        """
        await self.initialize()
        
        # Process in batches
        results = []
        total_batches = (len(test_cases) + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(0, len(test_cases), self.config.batch_size):
            batch = test_cases[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            
            if show_progress:
                logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Prepare batch data
            data = {
                "question": [tc.question for tc in batch],
                "answer": [tc.generated_answer or "" for tc in batch],
                "contexts": [tc.contexts for tc in batch],
                "ground_truth": [tc.ground_truth for tc in batch]
            }
            
            dataset = pd.DataFrame(data)
            
            try:
                # Run batch evaluation
                batch_result = evaluate(
                    dataset=dataset,
                    metrics=self._metrics,
                    llm=self._llm_wrapper,
                    embeddings=self._embeddings_wrapper,
                    raise_exceptions=False
                )
                
                # Convert results
                for idx, test_case in enumerate(batch):
                    metric_results = []
                    
                    # Extract metrics for this test case
                    for metric_name in batch_result.columns:
                        if metric_name not in ["question", "answer", "contexts", "ground_truth"]:
                            score = batch_result.iloc[idx][metric_name]
                            metric_results.append(MetricResult(
                                name=metric_name,
                                score=float(score) if not pd.isna(score) else 0.0,
                                confidence=0.9,
                                details={"ragas": True}
                            ))
                    
                    results.append(EvaluationResult(
                        test_case_id=test_case.test_case_id,
                        generated_answer=test_case.generated_answer,
                        metrics=metric_results,
                        metadata={"evaluator": "ragas", "batch": batch_num}
                    ))
                    
            except Exception as e:
                logger.error(f"Batch {batch_num} evaluation failed: {e}")
                # Add empty results for failed batch
                for test_case in batch:
                    results.append(EvaluationResult(
                        test_case_id=test_case.test_case_id,
                        generated_answer=test_case.generated_answer,
                        metrics=[],
                        metadata={"error": str(e), "evaluator": "ragas", "batch": batch_num}
                    ))
        
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate evaluation report with statistics.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Report dictionary
        """
        if not results:
            return {"error": "No results to analyze"}
        
        # Collect metrics
        metrics_data = {}
        for result in results:
            for metric in result.metrics:
                if metric.name not in metrics_data:
                    metrics_data[metric.name] = []
                metrics_data[metric.name].append(metric.score)
        
        # Calculate statistics
        report = {
            "total_evaluations": len(results),
            "successful_evaluations": sum(1 for r in results if r.metrics),
            "failed_evaluations": sum(1 for r in results if not r.metrics),
            "metrics": {}
        }
        
        for metric_name, scores in metrics_data.items():
            if scores:
                report["metrics"][metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "median": float(np.median(scores)),
                    "count": len(scores)
                }
        
        # Overall score
        all_scores = []
        for scores in metrics_data.values():
            all_scores.extend(scores)
        
        if all_scores:
            report["overall"] = {
                "mean": float(np.mean(all_scores)),
                "std": float(np.std(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores)),
                "median": float(np.median(all_scores))
            }
        
        return report


# Factory function for easy creation
async def create_ragas_evaluator(config: Union[RagasConfig, RAGConfig, Dict[str, Any]]) -> RagasEvaluator:
    """Create and initialize a Ragas evaluator.
    
    Args:
        config: Configuration (RagasConfig, RAGConfig, or dict)
        
    Returns:
        Initialized RagasEvaluator
    """
    # Convert to RagasConfig if needed
    if isinstance(config, dict):
        ragas_config = RagasConfig(**config)
    elif isinstance(config, RAGConfig):
        ragas_config = RagasConfig(
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            llm_api_key=config.llm_api_key,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_api_key=config.embedding_api_key
        )
    else:
        ragas_config = config
    
    evaluator = RagasEvaluator(ragas_config)
    await evaluator.initialize()
    return evaluator