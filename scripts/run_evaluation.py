"""
KnowledgeCore Engine 评测入口

这是所有评测的统一入口文件。
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from knowledge_core_engine import KnowledgeEngine
from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.evaluation.evaluator import Evaluator
from knowledge_core_engine.core.evaluation.golden_dataset import GoldenDataset
import logging

logger = logging.getLogger(__name__)


def calculate_keyword_coverage(generated_answer: str, ground_truth: str) -> float:
    """计算关键词覆盖率"""
    if not generated_answer or not ground_truth:
        return 0.0
    
    # 简单的实现：计算ground truth中的关键词在生成答案中的覆盖率
    import jieba
    
    # 提取关键词（这里用简单的分词）
    ground_truth_words = set(jieba.cut(ground_truth))
    generated_words = set(jieba.cut(generated_answer))
    
    # 过滤停用词（简单过滤）
    stop_words = {'的', '了', '是', '在', '和', '有', '与', '为', '等', '及', '或', '但', '而'}
    ground_truth_words = {w for w in ground_truth_words if len(w) > 1 and w not in stop_words}
    
    if not ground_truth_words:
        return 1.0
    
    # 计算覆盖率
    covered = ground_truth_words & generated_words
    return len(covered) / len(ground_truth_words)


def get_evaluation_config(profile: str = "default") -> RAGConfig:
    """获取评测配置
    
    Args:
        profile: 配置档案（目前只支持default）
    """
    # 直接使用默认配置
    return RAGConfig()


async def prepare_knowledge_base(engine: KnowledgeEngine, test_cases: list) -> None:
    """准备评测用的知识库
    
    从测试集中提取contexts，创建合成文档并加入知识库
    """
    print("准备知识库...")
    
    # 创建一个包含所有测试contexts的合成文档
    synthetic_content = []
    synthetic_content.append("# RAG系统知识库\n\n")
    
    for test_case in test_cases:
        synthetic_content.append(f"## {test_case.question}\n\n")
        for ctx in test_case.contexts:
            synthetic_content.append(f"{ctx}\n\n")
        synthetic_content.append("---\n\n")
    
    # 保存为临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(''.join(synthetic_content))
        temp_path = f.name
    
    # 添加到知识库
    await engine.add(temp_path)
    
    # 清理临时文件
    import os
    os.unlink(temp_path)
    
    print(f"  已添加包含 {len(test_cases)} 个测试案例相关内容的文档")


async def run_evaluation(
    config_profile: str = "default",
    dataset_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    metrics: Optional[list] = None,
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """运行评测
    
    Args:
        config_profile: 配置档案名称
        dataset_path: 测试集路径，默认使用内置测试集
        num_samples: 要测试的样本数，None表示全部
        metrics: 要评测的指标列表
        output_dir: 结果输出目录
    
    Returns:
        评测报告
    """
    print(f"\n{'='*60}")
    print(f"KnowledgeCore Engine 评测")
    print(f"配置档案: {config_profile}")
    print(f"{'='*60}\n")
    
    # 获取配置
    config = get_evaluation_config(config_profile)
    
    # 创建引擎
    print("1. 创建知识引擎...")
    engine = KnowledgeEngine(config=config, log_level="INFO")
    
    # 加载测试集
    print("2. 加载测试集...")
    if dataset_path is None:
        dataset_path = "data/golden_set/rag_qa_dataset.json"
    
    # 使用类方法加载数据集
    dataset = GoldenDataset.load(dataset_path)
    test_cases = dataset.test_cases
    
    if num_samples:
        test_cases = test_cases[:num_samples]
    
    print(f"   加载了 {len(test_cases)} 个测试用例")
    
    # 3. 准备知识库
    print("3. 准备知识库...")
    await prepare_knowledge_base(engine, test_cases)
    
    # 4. 生成答案
    print("4. 为测试用例生成答案...")
    for i, test_case in enumerate(test_cases):
        print(f"   处理 {i+1}/{len(test_cases)}: {test_case.question[:30]}...")
        try:
            # 使用RAG系统生成答案
            answer = await engine.ask(test_case.question)
            test_case.generated_answer = answer
        except Exception as e:
            logger.error(f"Failed to generate answer for {test_case.test_case_id}: {e}")
            test_case.generated_answer = ""
    
    # 设置评测指标
    if metrics is None:
        metrics = [
            "keyword_coverage",
            "response_time", 
            "answer_relevancy",
            "faithfulness",
            "context_precision"
        ]
    
    # 5. 运行评测
    print(f"5. 运行评测...")
    start_time = datetime.now()
    
    # 简单的评测实现（因为Evaluator接口有问题）
    results = []
    for test_case in test_cases:
        # 计算简单的指标
        result = {
            "test_case_id": test_case.test_case_id,
            "question": test_case.question,
            "generated_answer": test_case.generated_answer,
            "ground_truth": test_case.ground_truth,
            "metrics": {
                "answer_length": len(test_case.generated_answer) if test_case.generated_answer else 0,
                "has_answer": bool(test_case.generated_answer),
                # 简单的关键词覆盖率
                "keyword_coverage": calculate_keyword_coverage(
                    test_case.generated_answer, 
                    test_case.ground_truth
                ) if test_case.generated_answer else 0
            }
        }
        results.append(result)
    
    # 生成报告
    print("6. 生成评测报告...")
    evaluation_time = (datetime.now() - start_time).total_seconds()
    
    # 计算汇总统计
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r["metrics"]["has_answer"])
    avg_keyword_coverage = sum(r["metrics"]["keyword_coverage"] for r in results) / total_cases if total_cases > 0 else 0
    
    report = {
        "metadata": {
            "config_profile": config_profile,
            "dataset_path": dataset_path,
            "num_samples": len(test_cases),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": evaluation_time
        },
        "results": results,
        "summary": {
            "total_cases": total_cases,
            "successful_cases": successful_cases,
            "success_rate": successful_cases / total_cases if total_cases > 0 else 0,
            "avg_keyword_coverage": avg_keyword_coverage,
            "avg_answer_length": sum(r["metrics"]["answer_length"] for r in results) / total_cases if total_cases > 0 else 0
        }
    }
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"evaluation_{config_profile}_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n6. 评测完成！结果已保存到: {result_file}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("评测摘要")
    print(f"{'='*60}")
    print(f"配置档案: {config_profile}")
    print(f"测试样本数: {len(test_cases)}")
    print(f"总耗时: {report['metadata']['duration']:.2f}秒")
    
    summary = report['summary']
    print(f"\n基本指标:")
    print(f"  - 成功率: {summary['success_rate']:.2%}")
    print(f"  - 平均关键词覆盖率: {summary['avg_keyword_coverage']:.2%}")
    print(f"  - 平均答案长度: {summary['avg_answer_length']:.0f}字符")
    print(f"  - 评测耗时: {report['metadata']['duration']:.2f}秒")
    
    await engine.close()
    
    return report


async def compare_configs():
    """比较功能已移除"""
    print("配置比较功能已移除，请直接运行评测。")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="KnowledgeCore Engine 评测工具")
    parser.add_argument("--config", default="default", 
                      help="配置档案（目前只支持default）")
    parser.add_argument("--dataset", help="测试集路径")
    parser.add_argument("--samples", type=int, help="测试样本数")
    parser.add_argument("--metrics", nargs="+", help="评测指标")
    parser.add_argument("--compare", action="store_true", help="比较不同配置")
    parser.add_argument("--output", default="evaluation_results", help="输出目录")
    
    args = parser.parse_args()
    
    # 在解析参数后才运行异步代码
    async def async_main():
        if args.compare:
            await compare_configs()
        else:
            await run_evaluation(
                config_profile=args.config,
                dataset_path=args.dataset,
                num_samples=args.samples,
                metrics=args.metrics,
                output_dir=args.output
            )
    
    asyncio.run(async_main())


if __name__ == "__main__":
    main()