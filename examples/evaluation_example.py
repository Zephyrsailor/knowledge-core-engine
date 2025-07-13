"""Example of using the evaluation module."""

import asyncio
import os
from pathlib import Path
from knowledge_core_engine.core.evaluation import (
    Evaluator,
    EvaluationConfig,
    GoldenDataset,
    TestCase
)
from knowledge_core_engine.core.evaluation.report_generator import ReportGenerator
from knowledge_core_engine.core.evaluation.ragas_adapter import RagasAdapter


async def main():
    """Run evaluation example."""
    print("🔍 RAG System Evaluation Example")
    print("=" * 50)
    
    # 1. Load golden dataset
    dataset_path = Path("data/golden_set/rag_qa_dataset.json")
    if dataset_path.exists():
        print(f"\n📚 Loading golden dataset from {dataset_path}")
        dataset = GoldenDataset.load(dataset_path)
        print(f"   Loaded {len(dataset)} test cases")
    else:
        print("\n📝 Creating sample dataset")
        dataset = create_sample_dataset()
    
    # 2. Configure evaluation
    config = EvaluationConfig(
        metrics=["faithfulness", "answer_relevancy", "context_precision"],
        llm_provider="mock",  # Use mock for demo
        batch_size=5,
        verbose=True
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   Metrics: {', '.join(config.metrics)}")
    print(f"   LLM Provider: {config.llm_provider}")
    
    # 3. Simulate RAG responses for test cases
    print("\n🤖 Simulating RAG system responses...")
    for test_case in dataset:
        # In real usage, you would call your RAG system here
        test_case.generated_answer = simulate_rag_response(test_case)
    
    # 4. Run evaluation
    print("\n📊 Running evaluation...")
    evaluator = Evaluator(config)
    
    # Evaluate with custom evaluator
    results = await evaluator.evaluate_batch(
        list(dataset)[:5],  # Evaluate first 5 cases for demo
        show_progress=True
    )
    
    # 5. Try RAGAS integration
    print("\n🔧 Testing RAGAS integration...")
    ragas_adapter = RagasAdapter(use_ragas=False)  # Use mock
    ragas_results = await ragas_adapter.evaluate_with_ragas(
        list(dataset)[:3],
        config.metrics
    )
    
    print(f"   RAGAS evaluated {len(ragas_results)} test cases")
    
    # 6. Generate evaluation report
    print("\n📝 Generating evaluation report...")
    report_gen = ReportGenerator()
    
    # Generate different format reports
    reports_dir = Path("reports/evaluation")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Markdown report
    md_report = report_gen.generate_report(
        results,
        output_path=reports_dir / "evaluation_report.md",
        format="markdown",
        metadata={
            "dataset": dataset.name,
            "model": "mock",
            "date": "2024-01-15"
        }
    )
    print(f"   ✅ Markdown report saved")
    
    # HTML report
    html_report = report_gen.generate_report(
        results,
        output_path=reports_dir / "evaluation_report.html",
        format="html"
    )
    print(f"   ✅ HTML report saved")
    
    # JSON report
    json_report = report_gen.generate_report(
        results,
        output_path=reports_dir / "evaluation_report.json",
        format="json"
    )
    print(f"   ✅ JSON report saved")
    
    # 7. Display summary
    print("\n📈 Evaluation Summary:")
    summary = evaluator.generate_summary(results)
    print(f"   Overall Score: {summary['overall_score']:.3f}")
    print(f"   Average Scores:")
    for metric, score in summary['average_scores'].items():
        print(f"     - {metric}: {score:.3f}")
    
    print("\n✅ Evaluation complete!")


def create_sample_dataset() -> GoldenDataset:
    """Create a sample dataset for testing."""
    dataset = GoldenDataset("sample_dataset")
    
    # Add some test cases
    dataset.add_test_case(
        question="What is RAG?",
        ground_truth="RAG is Retrieval-Augmented Generation, combining retrieval and generation.",
        contexts=[
            "RAG combines information retrieval with text generation.",
            "Retrieval-Augmented Generation improves answer accuracy."
        ],
        metadata={"category": "definition", "difficulty": "easy"}
    )
    
    dataset.add_test_case(
        question="What are the benefits of RAG?",
        ground_truth="RAG benefits include accuracy, verifiability, and cost-effectiveness.",
        contexts=[
            "RAG provides accurate answers based on retrieved documents.",
            "Benefits include traceability and lower costs than fine-tuning."
        ],
        metadata={"category": "benefits", "difficulty": "medium"}
    )
    
    return dataset


def simulate_rag_response(test_case: TestCase) -> str:
    """Simulate a RAG system response."""
    # In real usage, this would call your actual RAG system
    # For demo, we'll create a simple response based on contexts
    
    if "definition" in test_case.metadata.get("category", ""):
        return "RAG is a technique that combines retrieval and generation for better AI responses."
    elif "benefits" in test_case.metadata.get("category", ""):
        return "The main benefits of RAG include improved accuracy and the ability to trace answers to sources."
    else:
        # Combine some words from contexts
        words = []
        for context in test_case.contexts[:2]:
            words.extend(context.split()[:10])
        return " ".join(words[:20]) + "..."


if __name__ == "__main__":
    asyncio.run(main())