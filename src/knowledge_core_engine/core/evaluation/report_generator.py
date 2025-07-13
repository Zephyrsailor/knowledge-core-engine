"""Evaluation report generator."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics

from .evaluator import EvaluationResult, MetricResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate evaluation reports in various formats."""
    
    def __init__(self):
        """Initialize report generator."""
        self.timestamp = datetime.now()
    
    def generate_report(self, results: List[EvaluationResult], 
                       output_path: Optional[Path] = None,
                       format: str = "markdown",
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate evaluation report.
        
        Args:
            results: List of evaluation results
            output_path: Optional path to save report
            format: Report format (markdown, html, json)
            metadata: Additional metadata for report
            
        Returns:
            Report content as string
        """
        if format == "markdown":
            report = self._generate_markdown_report(results, metadata)
        elif format == "html":
            report = self._generate_html_report(results, metadata)
        elif format == "json":
            report = self._generate_json_report(results, metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _generate_markdown_report(self, results: List[EvaluationResult],
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate Markdown format report."""
        lines = []
        
        # Header
        lines.append("# RAG System Evaluation Report")
        lines.append(f"\n**Generated**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if metadata:
            lines.append("\n## Configuration")
            for key, value in metadata.items():
                lines.append(f"- **{key}**: {value}")
        
        # Summary statistics
        lines.append("\n## Summary Statistics")
        summary = self._calculate_summary(results)
        
        lines.append(f"\n- **Total Test Cases**: {summary['total_cases']}")
        lines.append(f"- **Overall Score**: {summary['overall_score']:.3f}")
        
        lines.append("\n### Metric Averages")
        for metric, score in summary['metric_averages'].items():
            lines.append(f"- **{metric}**: {score:.3f}")
        
        # Score distribution
        lines.append("\n### Score Distribution")
        lines.append("\n| Metric | Min | Max | Mean | Std Dev |")
        lines.append("|--------|-----|-----|------|---------|")
        
        for metric, stats in summary['metric_stats'].items():
            lines.append(
                f"| {metric} | {stats['min']:.3f} | {stats['max']:.3f} | "
                f"{stats['mean']:.3f} | {stats['std']:.3f} |"
            )
        
        # Detailed results
        lines.append("\n## Detailed Results")
        
        for i, result in enumerate(results, 1):
            lines.append(f"\n### Test Case {i}: {result.test_case_id}")
            lines.append(f"\n**Generated Answer**: {result.generated_answer[:200]}...")
            
            lines.append("\n**Scores**:")
            for metric in result.metrics:
                lines.append(f"- {metric.name}: {metric.score:.3f}")
            
            if result.metadata:
                lines.append("\n**Metadata**:")
                for key, value in result.metadata.items():
                    lines.append(f"- {key}: {value}")
        
        # Performance insights
        lines.append("\n## Performance Insights")
        insights = self._generate_insights(results, summary)
        for insight in insights:
            lines.append(f"- {insight}")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, results: List[EvaluationResult],
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate HTML format report."""
        summary = self._calculate_summary(results)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .score {{ font-weight: bold; }}
        .good {{ color: green; }}
        .medium {{ color: orange; }}
        .poor {{ color: red; }}
        .metric-chart {{ margin: 20px 0; }}
        .summary-box {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>RAG System Evaluation Report</h1>
    <p><strong>Generated:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary-box">
        <h2>Summary</h2>
        <p><strong>Total Test Cases:</strong> {summary['total_cases']}</p>
        <p><strong>Overall Score:</strong> <span class="score {self._score_class(summary['overall_score'])}">{summary['overall_score']:.3f}</span></p>
    </div>
    
    <h2>Metric Performance</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Average Score</th>
            <th>Min</th>
            <th>Max</th>
            <th>Std Dev</th>
        </tr>
"""
        
        for metric, avg_score in summary['metric_averages'].items():
            stats = summary['metric_stats'][metric]
            html += f"""
        <tr>
            <td>{metric}</td>
            <td class="score {self._score_class(avg_score)}">{avg_score:.3f}</td>
            <td>{stats['min']:.3f}</td>
            <td>{stats['max']:.3f}</td>
            <td>{stats['std']:.3f}</td>
        </tr>
"""
        
        html += """
    </table>
    
    <h2>Detailed Results</h2>
"""
        
        for i, result in enumerate(results, 1):
            avg_score = result.average_score()
            html += f"""
    <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0;">
        <h3>Test Case {i}: {result.test_case_id}</h3>
        <p><strong>Average Score:</strong> <span class="score {self._score_class(avg_score)}">{avg_score:.3f}</span></p>
        <table style="width: auto;">
            <tr><th>Metric</th><th>Score</th></tr>
"""
            for metric in result.metrics:
                html += f"""
            <tr>
                <td>{metric.name}</td>
                <td class="score {self._score_class(metric.score)}">{metric.score:.3f}</td>
            </tr>
"""
            html += """
        </table>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def _generate_json_report(self, results: List[EvaluationResult],
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate JSON format report."""
        summary = self._calculate_summary(results)
        
        report_data = {
            "metadata": {
                "generated_at": self.timestamp.isoformat(),
                "total_test_cases": len(results),
                **(metadata or {})
            },
            "summary": summary,
            "results": [result.to_dict() for result in results],
            "insights": self._generate_insights(results, summary)
        }
        
        return json.dumps(report_data, ensure_ascii=False, indent=2)
    
    def _calculate_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not results:
            return {
                "total_cases": 0,
                "overall_score": 0.0,
                "metric_averages": {},
                "metric_stats": {}
            }
        
        # Collect scores by metric
        metric_scores: Dict[str, List[float]] = {}
        
        for result in results:
            for metric in result.metrics:
                if metric.name not in metric_scores:
                    metric_scores[metric.name] = []
                metric_scores[metric.name].append(metric.score)
        
        # Calculate statistics
        metric_averages = {}
        metric_stats = {}
        
        for metric_name, scores in metric_scores.items():
            metric_averages[metric_name] = statistics.mean(scores)
            metric_stats[metric_name] = {
                "min": min(scores),
                "max": max(scores),
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0
            }
        
        # Overall score
        all_scores = []
        for scores in metric_scores.values():
            all_scores.extend(scores)
        
        overall_score = statistics.mean(all_scores) if all_scores else 0.0
        
        return {
            "total_cases": len(results),
            "overall_score": overall_score,
            "metric_averages": metric_averages,
            "metric_stats": metric_stats
        }
    
    def _generate_insights(self, results: List[EvaluationResult], 
                          summary: Dict[str, Any]) -> List[str]:
        """Generate performance insights."""
        insights = []
        
        # Overall performance
        overall_score = summary['overall_score']
        if overall_score >= 0.85:
            insights.append("✅ Excellent overall performance (>85%)")
        elif overall_score >= 0.70:
            insights.append("⚡ Good overall performance (70-85%)")
        else:
            insights.append("⚠️ Performance needs improvement (<70%)")
        
        # Metric-specific insights
        for metric, avg_score in summary['metric_averages'].items():
            if avg_score < 0.60:
                insights.append(f"🔴 {metric} needs significant improvement (avg: {avg_score:.2f})")
            elif avg_score < 0.75:
                insights.append(f"🟡 {metric} has room for improvement (avg: {avg_score:.2f})")
        
        # Consistency analysis
        for metric, stats in summary['metric_stats'].items():
            if stats['std'] > 0.2:
                insights.append(f"📊 High variance in {metric} scores (std: {stats['std']:.2f})")
        
        # Failed cases
        failed_cases = sum(1 for r in results if r.average_score() < 0.5)
        if failed_cases > 0:
            insights.append(f"❌ {failed_cases} test case(s) performed poorly (<50%)")
        
        return insights
    
    def _score_class(self, score: float) -> str:
        """Get CSS class for score coloring."""
        if score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "medium"
        else:
            return "poor"