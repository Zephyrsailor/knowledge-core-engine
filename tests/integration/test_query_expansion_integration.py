"""Integration tests for query expansion functionality.

根据新的集成开发规则，确保查询扩展的每个查询都被独立使用。
"""

import pytest
import tempfile
from pathlib import Path

from knowledge_core_engine import KnowledgeEngine
from knowledge_core_engine.utils.logger import setup_logger

logger = setup_logger(__name__)


class TestQueryExpansionIntegration:
    """测试查询扩展的真实集成"""
    
    @pytest.mark.asyncio
    async def test_expanded_queries_used_independently(self):
        """测试扩展的查询被独立使用，而不是简单拼接"""
        engine = KnowledgeEngine(
            enable_query_expansion=True,
            query_expansion_method="rule_based",
            query_expansion_count=3,
            retrieval_strategy="vector",
            retrieval_top_k=5,
            extra_params={"debug_mode": True}
        )
        
        # 准备测试文档
        test_docs = [
            "RAG技术是检索增强生成的缩写",
            "检索增强生成是一种结合检索和生成的方法",
            "Retrieval Augmented Generation combines retrieval and generation",
            "这种方法可以提高生成内容的准确性",
            "技巧：使用向量数据库存储文档"
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, doc in enumerate(test_docs):
                file_path = Path(tmpdir) / f"doc{i}.txt"
                file_path.write_text(doc, encoding="utf-8")
            
            await engine.add(Path(tmpdir))
        
        # 执行查询，"RAG技术"应该扩展为多个查询
        results = await engine.ask("RAG技术", retrieval_only=True)
        
        # 验证结果
        assert len(results) > 0, "应该返回结果"
        
        # 检查是否找到了不同的相关文档
        # 如果查询扩展正确工作，应该找到：
        # 1. 包含"RAG"的文档
        # 2. 包含"技术"或其同义词的文档
        # 3. 包含"Retrieval Augmented Generation"的文档（如果扩展包含英文）
        
        contents = [r.content for r in results]
        
        # 至少应该找到包含原始查询词的文档
        assert any("RAG" in c for c in contents), "应该找到包含RAG的文档"
        
        # 应该也找到扩展查询相关的文档
        # 例如"技巧"（技术的同义词）或英文版本
        expanded_found = any(
            "技巧" in c or "Retrieval Augmented Generation" in c 
            for c in contents
        )
        assert expanded_found, "应该找到通过扩展查询发现的文档"
        
        logger.info(f"Found {len(results)} results with query expansion")
    
    @pytest.mark.asyncio
    async def test_query_expansion_improves_recall(self):
        """测试查询扩展提高召回率"""
        # 准备测试文档
        test_docs = [
            "什么是人工智能？人工智能是计算机科学的分支。",
            "AI（Artificial Intelligence）是模拟人类智能的技术。",
            "机器学习是人工智能的一个重要方法。",
            "深度学习是机器学习的一种技巧。",
            "神经网络是深度学习的基础技术。"
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, doc in enumerate(test_docs):
                file_path = Path(tmpdir) / f"doc{i}.txt"
                file_path.write_text(doc, encoding="utf-8")
            
            # 测试不使用查询扩展
            engine_no_expansion = KnowledgeEngine(
                enable_query_expansion=False,
                retrieval_strategy="vector",
                retrieval_top_k=10,
                persist_directory=str(Path(tmpdir) / "no_expansion_db")
            )
            await engine_no_expansion.add(Path(tmpdir))
            
            # 测试使用查询扩展
            engine_with_expansion = KnowledgeEngine(
                enable_query_expansion=True,
                query_expansion_method="rule_based",
                query_expansion_count=3,
                retrieval_strategy="vector",
                retrieval_top_k=10,
                persist_directory=str(Path(tmpdir) / "with_expansion_db")
            )
            await engine_with_expansion.add(Path(tmpdir))
        
        # 查询"什么是AI"
        query = "什么是AI"
        
        results_no_expansion = await engine_no_expansion.ask(query, retrieval_only=True)
        results_with_expansion = await engine_with_expansion.ask(query, retrieval_only=True)
        
        # 验证查询扩展找到更多相关文档
        # 不扩展可能只找到包含"AI"的文档
        # 扩展后应该也找到"人工智能"相关的文档
        assert len(results_with_expansion) >= len(results_no_expansion), \
            "查询扩展应该找到至少同样多的文档"
        
        # 检查是否找到了"人工智能"相关的文档
        contents_with_expansion = [r.content for r in results_with_expansion]
        ai_related_found = any("人工智能" in c for c in contents_with_expansion)
        assert ai_related_found, "查询扩展应该找到'人工智能'相关文档"
    
    @pytest.mark.asyncio
    async def test_query_expansion_merges_results_properly(self):
        """测试查询扩展正确合并结果，避免重复"""
        engine = KnowledgeEngine(
            enable_query_expansion=True,
            query_expansion_method="rule_based",
            query_expansion_count=3,
            retrieval_strategy="bm25",  # 使用BM25更容易看到效果
            retrieval_top_k=5,
            extra_params={"debug_mode": True}
        )
        
        # 准备测试文档
        test_docs = [
            "如何学习编程？首先要选择一门编程语言。",
            "怎么学习编程？可以从Python开始。",
            "怎样掌握编程技能？需要大量练习。",
            "编程学习方法包括看书、做项目、参加课程。",
            "学习编程的技巧是保持耐心和持续练习。"
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, doc in enumerate(test_docs):
                file_path = Path(tmpdir) / f"doc{i}.txt"
                file_path.write_text(doc, encoding="utf-8")
            
            await engine.add(Path(tmpdir))
        
        # 查询"如何学习编程"
        # 应该扩展为：如何学习编程、怎么学习编程、怎样学习编程等
        results = await engine.ask("如何学习编程", retrieval_only=True)
        
        # 验证结果
        assert len(results) > 0, "应该返回结果"
        
        # 检查结果不重复
        seen_ids = set()
        for r in results:
            assert r.chunk_id not in seen_ids, f"结果不应该重复: {r.chunk_id}"
            seen_ids.add(r.chunk_id)
        
        # 应该找到多个相关文档
        contents = [r.content for r in results]
        
        # 应该找到不同表达方式的文档
        found_variations = sum([
            any("如何" in c for c in contents),
            any("怎么" in c for c in contents),
            any("怎样" in c for c in contents),
            any("方法" in c for c in contents),
            any("技巧" in c for c in contents)
        ])
        
        assert found_variations >= 2, "应该找到多种表达方式的相关文档"
        
        logger.info(f"Found {len(results)} unique results from expanded queries")


@pytest.mark.asyncio
async def test_query_expansion_monitoring():
    """测试查询扩展的监控和日志"""
    engine = KnowledgeEngine(
        enable_query_expansion=True,
        query_expansion_method="rule_based",
        query_expansion_count=3,
        retrieval_strategy="vector",
        extra_params={
            "debug_mode": True,
            "log_query_expansion": True  # 启用详细日志
        }
    )
    
    # 简单测试以验证日志
    with tempfile.TemporaryDirectory() as tmpdir:
        doc = Path(tmpdir) / "test.txt"
        doc.write_text("测试文档", encoding="utf-8")
        await engine.add(doc)
    
    # 执行查询
    await engine.ask("什么是RAG", retrieval_only=True)
    
    # 在实际运行时，应该在日志中看到：
    # 1. "Query expanded from '什么是RAG' to X variations"
    # 2. 每个扩展查询的独立检索结果
    # 3. 结果合并的信息


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])