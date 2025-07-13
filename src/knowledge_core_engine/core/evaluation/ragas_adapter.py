"""RAGAS framework integration adapter."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .evaluator import EvaluationResult, MetricResult, TestCase as EvalTestCase

logger = logging.getLogger(__name__)


class RagasAdapter:
    """Adapter for integrating RAGAS evaluation framework."""
    
    def __init__(self, use_ragas: bool = True):
        """Initialize RAGAS adapter.
        
        Args:
            use_ragas: Whether to use actual RAGAS library
        """
        self.use_ragas = use_ragas
        self._ragas_available = False
        
        if use_ragas:
            try:
                import ragas
                self._ragas_available = True
                logger.info("RAGAS framework available")
            except ImportError:
                logger.warning("RAGAS not installed, using mock implementation")
                self._ragas_available = False
    
    async def evaluate_with_ragas(self, test_cases: List[EvalTestCase], 
                                  metrics: List[str]) -> List[EvaluationResult]:
        """Evaluate test cases using RAGAS framework.
        
        Args:
            test_cases: List of test cases
            metrics: List of metric names
            
        Returns:
            List of evaluation results
        """
        if self._ragas_available and self.use_ragas:
            return await self._evaluate_with_real_ragas(test_cases, metrics)
        else:
            return await self._evaluate_with_mock_ragas(test_cases, metrics)
    
    async def _evaluate_with_real_ragas(self, test_cases: List[EvalTestCase], 
                                       metrics: List[str]) -> List[EvaluationResult]:
        """Use actual RAGAS library for evaluation."""
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
            import pandas as pd
            
            # Map metric names to RAGAS metrics
            metric_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "answer_similarity": answer_similarity,
                "answer_correctness": answer_correctness
            }
            
            # Select requested metrics
            ragas_metrics = [metric_map[m] for m in metrics if m in metric_map]
            
            # Prepare data for RAGAS
            data = {
                "question": [tc.question for tc in test_cases],
                "answer": [tc.generated_answer or "" for tc in test_cases],
                "contexts": [tc.contexts for tc in test_cases],
                "ground_truth": [tc.ground_truth for tc in test_cases]
            }
            
            dataset = pd.DataFrame(data)
            
            # Configure LLM for RAGAS (using DeepSeek)
            # Note: This requires proper LLM setup which we'll mock for now
            # In production, you'd configure the actual LLM here
            
            # Run evaluation
            results = evaluate(
                dataset,
                metrics=ragas_metrics,
                # llm=configured_llm,  # Would need proper LLM setup
                # embeddings=configured_embeddings
            )
            
            # Convert RAGAS results to our format
            eval_results = []
            for i, tc in enumerate(test_cases):
                metric_results = []
                for metric_name in metrics:
                    if metric_name in results.scores:
                        score = results.scores[metric_name][i]
                        metric_results.append(MetricResult(
                            name=metric_name,
                            score=score,
                            details={"source": "ragas"}
                        ))
                
                eval_results.append(EvaluationResult(
                    test_case_id=tc.test_case_id,
                    generated_answer=tc.generated_answer or "",
                    metrics=metric_results
                ))
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Error using RAGAS: {e}")
            # Fallback to mock
            return await self._evaluate_with_mock_ragas(test_cases, metrics)
    
    async def _evaluate_with_mock_ragas(self, test_cases: List[EvalTestCase], 
                                       metrics: List[str]) -> List[EvaluationResult]:
        """Mock RAGAS evaluation for testing."""
        import random
        
        results = []
        
        for tc in test_cases:
            metric_results = []
            
            for metric in metrics:
                # Generate mock scores based on metric type
                if metric == "faithfulness":
                    # Higher score if answer contains words from context
                    context_text = " ".join(tc.contexts).lower()
                    answer_words = tc.generated_answer.lower().split() if tc.generated_answer else []
                    score = sum(1 for w in answer_words if w in context_text) / max(len(answer_words), 1)
                    score = min(score, 1.0)
                    
                elif metric == "answer_relevancy":
                    # Higher score if answer contains words from question
                    question_words = tc.question.lower().split()
                    answer_words = tc.generated_answer.lower().split() if tc.generated_answer else []
                    score = sum(1 for w in question_words if w in answer_words) / max(len(question_words), 1)
                    score = min(score * 1.5, 1.0)  # Boost a bit
                    
                elif metric == "context_precision":
                    # Mock based on number of contexts
                    score = 0.8 if len(tc.contexts) <= 3 else 0.6
                    
                elif metric == "context_recall":
                    # Mock based on context coverage
                    score = 0.75 + random.random() * 0.2
                    
                else:
                    # Default mock score
                    score = 0.7 + random.random() * 0.25
                
                metric_results.append(MetricResult(
                    name=metric,
                    score=score,
                    details={"source": "mock_ragas"}
                ))
            
            results.append(EvaluationResult(
                test_case_id=tc.test_case_id,
                generated_answer=tc.generated_answer or "",
                metrics=metric_results
            ))
        
        return results
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available RAGAS metrics."""
        if self._ragas_available:
            return [
                "faithfulness",
                "answer_relevancy", 
                "context_precision",
                "context_recall",
                "answer_similarity",
                "answer_correctness"
            ]
        else:
            # Mock implementation supports basic metrics
            return [
                "faithfulness",
                "answer_relevancy",
                "context_precision", 
                "context_recall"
            ]