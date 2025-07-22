"""Integration tests for the complete RAG pipeline."""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch, AsyncMock, MagicMock

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.parsing.document_processor import DocumentProcessor
from knowledge_core_engine.core.chunking.pipeline import ChunkingPipeline
from knowledge_core_engine.core.enhancement.metadata_enhancer import MetadataEnhancer
from knowledge_core_engine.core.embedding.embedder import TextEmbedder
from knowledge_core_engine.core.embedding.vector_store import VectorStore
from knowledge_core_engine.core.retrieval.retriever import Retriever
from knowledge_core_engine.core.retrieval.reranker_wrapper import Reranker
from knowledge_core_engine.core.generation.generator import Generator


@pytest.mark.integration
class TestFullRAGPipeline:
    """Test the complete RAG pipeline from document to answer."""
    
    @pytest.fixture
    def config(self):
        """Create config for full pipeline."""
        return RAGConfig(
            # LLM settings
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            llm_api_key=os.getenv("DEEPSEEK_API_KEY", "test-key"),
            
            # Embedding settings
            embedding_provider="dashscope",
            embedding_model="text-embedding-v3",
            embedding_api_key=os.getenv("DASHSCOPE_API_KEY", "test-key"),
            
            # Vector DB settings
            vectordb_provider="chromadb",
            collection_name="test_rag_pipeline",
            persist_directory=tempfile.mkdtemp(),
            
            # Retrieval settings
            retrieval_strategy="hybrid",
            retrieval_top_k=5,
            reranker_model="bge-reranker-v2-m3-qwen",
            
            # Generation settings
            temperature=0.1,
            include_citations=True,
            
            # Features
            use_multi_vector=True
        )
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        content = """
        # 企业知识管理系统实施指南
        
        ## 第一章：知识管理概述
        
        知识管理是组织对其智力资产进行系统化管理的过程。在数字化时代，
        有效的知识管理已成为企业保持竞争优势的关键因素。
        
        ### 1.1 知识管理的定义
        
        知识管理（Knowledge Management, KM）是指通过对组织内外部知识资源的
        识别、获取、开发、分解、存储、传递、共享和评价，从而为组织创造价值的过程。
        
        ### 1.2 知识管理的重要性
        
        1. **提高决策质量**：通过整合分散的知识资源，为决策提供全面支持
        2. **加速创新**：促进知识共享和协作，激发创新思维
        3. **降低成本**：避免重复工作，提高工作效率
        4. **保护知识资产**：防止关键知识随人员流动而流失
        
        ## 第二章：RAG技术在知识管理中的应用
        
        ### 2.1 什么是RAG
        
        RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的
        人工智能技术。它通过从知识库中检索相关信息来增强语言模型的生成能力。
        
        ### 2.2 RAG的优势
        
        - **准确性高**：基于实际文档生成答案，减少虚假信息
        - **可追溯性**：每个答案都可以追溯到具体的源文档
        - **易于更新**：只需更新知识库，无需重新训练模型
        - **成本效益**：相比微调大模型，实施成本更低
        
        ### 2.3 RAG系统架构
        
        一个完整的RAG系统通常包括以下组件：
        1. 文档处理模块：负责解析和预处理各种格式的文档
        2. 向量化模块：将文本转换为向量表示
        3. 检索模块：从向量数据库中查找相关内容
        4. 生成模块：基于检索结果生成答案
        
        ## 第三章：实施步骤
        
        ### 3.1 需求分析
        
        在实施知识管理系统前，需要明确：
        - 目标用户群体
        - 知识类型和来源
        - 使用场景和频率
        - 性能和准确性要求
        
        ### 3.2 技术选型
        
        选择合适的技术栈：
        - 文档解析：LlamaParse、Apache Tika
        - 向量数据库：ChromaDB、Pinecone、Weaviate
        - 嵌入模型：OpenAI、Cohere、国产模型
        - 语言模型：GPT-4、Claude、DeepSeek、通义千问
        
        ### 3.3 实施流程
        
        1. **数据准备**：收集和整理知识文档
        2. **系统搭建**：部署RAG系统各组件
        3. **数据导入**：将文档导入系统
        4. **测试优化**：测试系统性能并优化
        5. **上线运营**：正式投入使用
        
        ## 第四章：最佳实践
        
        ### 4.1 文档管理最佳实践
        
        - 建立统一的文档命名规范
        - 定期更新和审核文档内容
        - 建立文档版本控制机制
        - 设置合适的访问权限
        
        ### 4.2 系统优化建议
        
        - 定期评估检索质量
        - 根据用户反馈调整参数
        - 监控系统性能指标
        - 建立知识更新流程
        
        ## 结语
        
        企业知识管理系统的成功实施需要技术和管理的紧密配合。
        通过采用RAG等先进技术，企业可以更好地利用知识资产，
        提升组织的整体竞争力。
        """
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            return Path(f.name)
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Test needs refactoring for new API")
    async def test_end_to_end_pipeline(self, config, sample_document):
        """Test complete pipeline from document to answer."""
        # Step 1: Parse document
        parser = DocumentProcessor()
        
        # Process document
        parse_result = await parser.process(sample_document)
        assert "企业知识管理系统实施指南" in parse_result.markdown
        markdown_content = parse_result.markdown
        metadata = parse_result.metadata
        
        # Step 2: Chunk document
        chunker = ChunkingPipeline()
        chunk_result = await chunker.process_parse_result(parse_result)
        chunks = chunk_result.chunks
        
        assert len(chunks) > 5  # Should create multiple chunks
        assert any("RAG" in chunk.content for chunk in chunks)
        
        # Step 3: Enhance metadata (mock LLM calls)
        from knowledge_core_engine.core.enhancement.metadata_enhancer import EnhancementConfig
        enhance_config = EnhancementConfig(llm_provider="mock")
        enhancer = MetadataEnhancer(enhance_config)
        
        with patch.object(enhancer, 'enhance_chunk') as mock_enhance:
            async def mock_enhancement(chunk):
                # Simulate enhancement
                if "RAG" in chunk.content:
                    chunk.metadata.update({
                        "summary": "介绍RAG技术及其优势",
                        "keywords": ["RAG", "检索增强生成", "知识管理"],
                        "questions": [
                            "什么是RAG技术？",
                            "RAG有哪些优势？",
                            "RAG如何应用于知识管理？"
                        ],
                        "chunk_type": "technical_explanation"
                    })
                return chunk
            
            mock_enhance.side_effect = mock_enhancement
            enhanced_chunks = await enhancer.enhance_batch(chunks[:3])  # Enhance first 3
        
        # Step 4: Embed and store (mock embedding calls)
        embedder = TextEmbedder(config)
        vector_store = VectorStore(config)
        
        await embedder.initialize()
        await vector_store.initialize()
        
        # Mock embedder's embed method
        with patch.object(embedder, 'embed') as mock_embed:
            # Mock embedding result
            mock_embed.return_value = MagicMock(embedding=[0.1] * 1536)
            
            # Store chunks
            for chunk in enhanced_chunks:
                embedding = await embedder.embed(chunk.content)
                await vector_store.add_documents([{
                    "id": chunk.chunk_id,
                    "content": chunk.content,
                    "embedding": embedding.embedding,
                    "metadata": chunk.metadata
                }])
        
        # Step 5: Retrieve relevant chunks
        retriever = Retriever(config)
        
        with patch.object(retriever, '_embedder', embedder), \
             patch.object(retriever, '_vector_store', vector_store):
            
            retriever._initialized = True
            
            # Mock vector search
            with patch.object(vector_store, 'query') as mock_query:
                mock_query.return_value = [
                    {
                        "id": enhanced_chunks[0].chunk_id,
                        "score": 0.95,
                        "content": enhanced_chunks[0].content,
                        "metadata": enhanced_chunks[0].metadata
                    }
                ]
                
                query = "什么是RAG技术？它有哪些优势？"
                results = await retriever.retrieve(query, top_k=3)
                
                assert len(results) > 0
                assert any("RAG" in r.content for r in results)
        
        # Step 6: Rerank results (optional)
        if results and len(results) > 1:
            reranker = Reranker(config)
            
            with patch.object(reranker, '_provider') as mock_provider:
                mock_provider.rerank = AsyncMock(return_value=[
                    {"index": 0, "score": 0.98}
                ])
                
                reranker._initialized = True
                reranked = await reranker.rerank(query, results, top_k=2)
                results = reranked or results
        
        # Step 7: Generate answer
        generator = Generator(config)
        
        with patch.object(generator, '_call_llm') as mock_llm:
            mock_llm.return_value = {
                "content": """
                根据文档内容，RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的人工智能技术[1]。
                
                RAG的主要优势包括[1]：
                1. **准确性高**：基于实际文档生成答案，减少虚假信息
                2. **可追溯性**：每个答案都可以追溯到具体的源文档
                3. **易于更新**：只需更新知识库，无需重新训练模型
                4. **成本效益**：相比微调大模型，实施成本更低
                
                在知识管理中，RAG技术可以帮助企业更有效地利用知识资产，提供准确可靠的信息检索和问答服务[1]。
                """.strip(),
                "usage": {"total_tokens": 500}
            }
            
            generator._initialized = True
            final_answer = await generator.generate(query, results)
            
            assert final_answer.answer
            assert "Retrieval-Augmented Generation" in final_answer.answer
            assert "准确性高" in final_answer.answer
            assert "[1]" in final_answer.answer
        
        # Cleanup
        os.unlink(sample_document)
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Test needs refactoring for new API")
    async def test_pipeline_with_multiple_queries(self, config):
        """Test pipeline handling multiple queries."""
        # Setup mock components
        retriever = Retriever(config)
        generator = Generator(config)
        
        queries = [
            "什么是知识管理？",
            "RAG技术如何工作？",
            "实施知识管理系统的步骤是什么？"
        ]
        
        results = []
        
        for query in queries:
            # Mock retrieval
            with patch.object(retriever, 'retrieve') as mock_retrieve:
                mock_retrieve.return_value = [
                    {
                        "chunk_id": f"chunk_{i}",
                        "content": f"关于{query}的相关内容...",
                        "score": 0.9 - i * 0.1,
                        "metadata": {"source": "test"}
                    }
                    for i in range(3)
                ]
                
                retriever._initialized = True
                contexts = await retriever.retrieve(query)
            
            # Mock generation
            with patch.object(generator, '_call_llm') as mock_llm:
                mock_llm.return_value = {
                    "content": f"这是关于'{query}'的答案。",
                    "usage": {"total_tokens": 200}
                }
                
                generator._initialized = True
                answer = await generator.generate(query, contexts)
                results.append(answer)
        
        assert len(results) == 3
        assert all(r.answer for r in results)
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Test needs refactoring for new API")
    async def test_pipeline_error_handling(self, config):
        """Test pipeline handles errors gracefully."""
        # Test parsing error
        parser = DocumentParser(config)
        
        with patch.object(parser, '_parse_with_llama_parse') as mock_parse:
            mock_parse.side_effect = Exception("Parsing failed")
            
            with pytest.raises(Exception, match="Parsing failed"):
                await parser.parse(Path("nonexistent.pdf"))
        
        # Test retrieval error
        retriever = Retriever(config)
        
        with patch.object(retriever, '_vector_store') as mock_store:
            mock_store.query.side_effect = Exception("Vector DB error")
            
            retriever._initialized = True
            retriever._embedder = AsyncMock()
            
            with pytest.raises(Exception, match="Vector DB error"):
                await retriever.retrieve("test query")
        
        # Test generation error with retry
        generator = Generator(config)
        generator.config.extra_params["max_retries"] = 2
        
        call_count = 0
        
        async def flaky_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary error")
            return {"content": "Success", "usage": {"total_tokens": 50}}
        
        generator._call_llm = flaky_llm
        generator._initialized = True
        
        result = await generator.generate("test", [])
        assert result.answer == "Success"
        assert call_count == 2


@pytest.mark.integration 
@pytest.mark.skip(reason="Test class needs refactoring for new API")
class TestRAGPipelineOptimization:
    """Test pipeline optimization features."""
    
    @pytest.mark.asyncio
    async def test_caching_optimization(self, config):
        """Test caching reduces redundant operations."""
        embedder = TextEmbedder(config)
        
        call_count = 0
        original_embed = None
        
        async def counting_embed(text):
            nonlocal call_count, original_embed
            call_count += 1
            if original_embed:
                return await original_embed(text)
            return {"embedding": [0.1] * 1536, "usage": {"tokens": 10}}
        
        with patch.object(embedder, '_embed_with_dashscope', counting_embed):
            await embedder.initialize()
            
            # First call
            result1 = await embedder.embed_text("test text")
            assert call_count == 1
            
            # Second call with same text (should use cache)
            result2 = await embedder.embed_text("test text")
            assert call_count == 1  # No additional call
            assert result1.embedding == result2.embedding
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, config):
        """Test batch processing is more efficient."""
        enhancer = MetadataEnhancer(config)
        
        # Create many chunks
        chunks = [
            {"id": f"chunk_{i}", "text": f"Test content {i}", "metadata": {}}
            for i in range(20)
        ]
        
        call_count = 0
        
        async def mock_llm_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"enhanced": True}
        
        with patch.object(enhancer, '_call_llm', mock_llm_call):
            # Process in batch
            await enhancer.enhance_batch(chunks, batch_size=5)
            
            # Should make 4 calls (20 chunks / 5 per batch)
            assert call_count == 4
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, config):
        """Test concurrent processing of multiple documents."""
        parser = DocumentParser(config)
        
        # Create multiple documents
        docs = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Document {i} content")
                docs.append(Path(f.name))
        
        try:
            with patch.object(parser, '_parse_with_llama_parse') as mock_parse:
                async def slow_parse(path):
                    await asyncio.sleep(0.1)  # Simulate slow parsing
                    return (f"Parsed {path.name}", {"pages": 1})
                
                mock_parse.side_effect = slow_parse
                
                # Parse concurrently
                import time
                start = time.time()
                
                tasks = [parser.parse(doc) for doc in docs]
                results = await asyncio.gather(*tasks)
                
                elapsed = time.time() - start
                
                # Should be faster than sequential (0.3s)
                assert elapsed < 0.2
                assert len(results) == 3
                
        finally:
            # Cleanup
            for doc in docs:
                os.unlink(doc)