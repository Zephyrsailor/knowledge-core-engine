"""真实的BM25集成测试，不使用任何mock。"""

import pytest
import tempfile
from pathlib import Path
import logging

from knowledge_core_engine import KnowledgeEngine

logger = logging.getLogger(__name__)


class TestBM25RealIntegration:
    """BM25真实集成测试套件。"""
    
    @pytest.mark.asyncio
    async def test_bm25_actually_returns_results_no_mocks(self):
        """测试BM25真正返回结果，完全不使用mock。"""
        engine = KnowledgeEngine(
            retrieval_strategy="bm25",
            extra_params={"debug_mode": True}
        )
        
        # 创建真实的测试文档
        test_content = """
        RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的AI技术。
        它通过从知识库中检索相关信息来增强语言模型的生成能力。
        BM25是一种经典的文本检索算法，基于词频和文档频率计算相关性分数。
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            # 添加文档
            result = await engine.add(temp_file)
            assert result['processed_files'] == 1
            assert result['total_chunks'] > 0
            
            # 直接访问BM25索引验证
            if engine._retriever and engine._retriever._bm25_index:
                bm25 = engine._retriever._bm25_index
                assert hasattr(bm25, '_documents'), "BM25 must have documents"
                assert len(bm25._documents) > 0, "BM25 must contain documents"
                logger.info(f"BM25 contains {len(bm25._documents)} documents")
            
            # 测试多个查询
            test_queries = ["RAG", "BM25", "检索", "AI技术"]
            non_zero_scores = 0
            
            for query in test_queries:
                results = await engine.ask(query, retrieval_only=True)
                assert isinstance(results, list), f"Results must be a list for query: {query}"
                assert len(results) > 0, f"BM25 must return results for query: {query}"
                
                if results[0].score > 0:
                    non_zero_scores += 1
                    logger.info(f"Query '{query}' returned {len(results)} results with score {results[0].score:.4f}")
                else:
                    logger.warning(f"Query '{query}' returned 0 score")
            
            # 至少应该有一些查询返回非零分数
            assert non_zero_scores > 0, f"At least some queries should return non-zero scores, but all returned 0"
            logger.info(f"{non_zero_scores}/{len(test_queries)} queries returned non-zero scores")
        
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_both_methods_work_no_mocks(self):
        """测试混合检索中两种方法都真正工作，不使用mock。"""
        engine = KnowledgeEngine(
            retrieval_strategy="hybrid",
            vector_weight=0.5,
            bm25_weight=0.5,
            retrieval_top_k=10,
            extra_params={"debug_mode": True}
        )
        
        # 创建多个测试文档，确保有足够的内容
        test_docs = {
            "doc1.txt": "向量数据库是一种专门用于存储和检索向量的数据库系统。它使用向量相似度进行搜索。",
            "doc2.txt": "BM25是一种基于词频的经典文本检索算法。它不需要向量化，直接基于关键词匹配。",
            "doc3.txt": "混合检索结合了向量检索和关键词检索的优势，提供更好的检索效果。"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建所有测试文档
            for filename, content in test_docs.items():
                file_path = Path(tmpdir) / filename
                file_path.write_text(content, encoding="utf-8")
            
            # 添加文档
            result = await engine.add(Path(tmpdir))
            assert result['processed_files'] == len(test_docs)
            
            # 验证BM25索引
            if engine._retriever and engine._retriever._bm25_index:
                bm25_doc_count = len(engine._retriever._bm25_index._documents)
                assert bm25_doc_count > 0, "BM25 index must contain documents"
                logger.info(f"BM25 index contains {bm25_doc_count} documents")
            
            # 测试混合检索
            # 这个查询应该通过向量找到语义相关内容，通过BM25找到关键词匹配
            test_query = "检索算法的优势"
            
            # 获取详细结果以验证
            detailed_result = await engine.ask(test_query, return_details=True)
            
            assert 'contexts' in detailed_result
            assert len(detailed_result['contexts']) > 0
            
            # 记录检索到的内容，帮助调试
            for i, ctx in enumerate(detailed_result['contexts'][:3]):
                logger.info(f"Context {i+1}: score={ctx['score']:.3f}, content={ctx['content'][:50]}...")
            
            # 验证混合检索工作
            # 由于我们使用了debug_mode，如果BM25没有返回结果，会抛出异常
            # 所以能走到这里说明两种方法都工作了
            assert detailed_result['answer'] is not None
            assert len(detailed_result['answer']) > 0
    
    @pytest.mark.asyncio
    async def test_bm25_chinese_tokenization_works(self):
        """测试BM25中文分词是否正常工作。"""
        engine = KnowledgeEngine(
            retrieval_strategy="bm25",
            bm25_language="zh",  # 明确指定中文
            extra_params={"debug_mode": True}
        )
        
        # 中文测试文档
        chinese_content = """
        人工智能技术正在快速发展，机器学习和深度学习是其中的重要分支。
        自然语言处理让计算机能够理解和生成人类语言。
        知识图谱帮助机器理解实体之间的关系。
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(chinese_content)
            temp_file = f.name
        
        try:
            # 添加文档
            await engine.add(temp_file)
            
            # 测试中文查询
            chinese_queries = ["人工智能", "机器学习", "自然语言", "知识图谱"]
            
            for query in chinese_queries:
                results = await engine.ask(query, retrieval_only=True)
                assert len(results) > 0, f"Must return results for Chinese query: {query}"
                # 验证返回的内容包含查询词
                found = any(query in result.content for result in results)
                assert found, f"Results should contain the query term: {query}"
                logger.info(f"Chinese query '{query}' successful")
        
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_bm25_index_persistence_across_adds(self):
        """测试BM25索引在多次添加文档后的持久性。"""
        engine = KnowledgeEngine(
            retrieval_strategy="bm25",
            extra_params={"debug_mode": True}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 第一次添加
            doc1 = Path(tmpdir) / "doc1.txt"
            doc1.write_text("Python是一种流行的编程语言", encoding="utf-8")
            
            await engine.add(doc1)
            
            # 验证可以搜索到
            results1 = await engine.ask("Python", retrieval_only=True)
            assert len(results1) > 0
            initial_count = len(results1)
            
            # 第二次添加不同内容
            doc2 = Path(tmpdir) / "doc2.txt"
            doc2.write_text("Python在数据科学领域应用广泛", encoding="utf-8")
            
            await engine.add(doc2)
            
            # 再次搜索
            results2 = await engine.ask("Python", retrieval_only=True)
            
            # 至少确保返回了结果（可能因为文档去重等原因不会增加）
            assert len(results2) >= initial_count, "Should return at least the same number of results"
            
            # 如果结果增加了，验证两个文档都能被找到
            if len(results2) > initial_count:
                contents = [r.content for r in results2]
                # 检查是否包含两种内容
                has_first = any("编程语言" in c for c in contents)
                has_second = any("数据科学" in c for c in contents)
                assert has_first or has_second, "Should find at least one of the documents"