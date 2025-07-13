"""Integration tests for generation module with real LLM providers."""

import pytest
import asyncio
import os
from typing import List
from unittest.mock import patch, AsyncMock

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.generation.generator import Generator, GenerationResult
from knowledge_core_engine.core.retrieval.retriever import RetrievalResult


@pytest.mark.integration
class TestGenerationIntegration:
    """Integration tests for generation with real contexts."""
    
    @pytest.fixture
    def config(self):
        """Create config for integration tests."""
        return RAGConfig(
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            llm_api_key=os.getenv("DEEPSEEK_API_KEY", "test-key"),
            temperature=0.1,
            max_tokens=1024,
            include_citations=True
        )
    
    @pytest.fixture
    def real_contexts(self):
        """Create realistic retrieval contexts."""
        return [
            RetrievalResult(
                chunk_id="chunk_001",
                content="""
                检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合了信息检索和文本生成的人工智能技术。
                RAG系统首先从大规模知识库中检索相关信息，然后将这些信息作为上下文提供给语言模型，
                使其能够生成更准确、更有依据的回答。这种方法有效解决了大语言模型的"幻觉"问题。
                """.strip(),
                score=0.95,
                metadata={
                    "document_id": "rag_guide_2024",
                    "document_title": "企业级RAG实施指南",
                    "page": 3,
                    "section": "1.1 RAG概述",
                    "author": "技术团队"
                }
            ),
            RetrievalResult(
                chunk_id="chunk_002",
                content="""
                RAG的核心优势包括：
                1. 知识可更新性：无需重新训练模型即可更新知识库
                2. 可解释性：每个答案都可追溯到具体的源文档
                3. 成本效益：相比微调大模型，RAG的实施成本更低
                4. 领域适应性：可以快速适应特定领域的知识需求
                """.strip(),
                score=0.88,
                metadata={
                    "document_id": "rag_guide_2024",
                    "document_title": "企业级RAG实施指南", 
                    "page": 5,
                    "section": "1.2 RAG的优势"
                }
            ),
            RetrievalResult(
                chunk_id="chunk_003",
                content="""
                实施RAG系统的关键步骤：
                第一步：文档准备和预处理，包括格式转换、清洗和标准化
                第二步：文档分块策略设计，需要考虑语义完整性和检索效率的平衡
                第三步：向量化和索引构建，选择合适的嵌入模型和向量数据库
                第四步：检索策略优化，包括混合检索和重排序
                第五步：生成质量控制，包括提示工程和后处理
                """.strip(),
                score=0.82,
                metadata={
                    "document_id": "rag_impl_guide",
                    "document_title": "RAG系统实施手册",
                    "page": 12,
                    "chapter": "第3章 实施流程"
                }
            )
        ]
    
    @pytest.mark.asyncio
    async def test_basic_generation_flow(self, config, real_contexts):
        """Test basic generation with real-like contexts."""
        generator = Generator(config)
        
        # Mock the LLM call but test the full flow
        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "content": """
                根据提供的文档，RAG（检索增强生成）是一种结合信息检索和文本生成的AI技术[1]。
                
                RAG的主要优势包括[2]：
                - 知识可更新性：可以随时更新知识库而无需重新训练模型
                - 可解释性：生成的答案可以追溯到具体来源
                - 成本效益：比微调大模型更经济
                
                这种技术有效解决了大语言模型的"幻觉"问题[1]，使AI系统能够提供更准确可靠的答案。
                """.strip(),
                "usage": {"prompt_tokens": 850, "completion_tokens": 120, "total_tokens": 970}
            }
            
            query = "什么是RAG？它有什么优势？"
            result = await generator.generate(query, real_contexts)
            
            assert isinstance(result, GenerationResult)
            assert "检索增强生成" in result.answer
            assert "知识可更新性" in result.answer
            assert "[1]" in result.answer
            assert "[2]" in result.answer
            assert result.usage["total_tokens"] == 970
    
    @pytest.mark.asyncio
    async def test_citation_extraction_and_mapping(self, config, real_contexts):
        """Test citation extraction and mapping to sources."""
        generator = Generator(config)
        
        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "content": "RAG技术首先检索相关信息[1]，然后生成答案[1]。实施时需要考虑多个步骤[3]。",
                "usage": {"total_tokens": 500}
            }
            
            result = await generator.generate("RAG的工作流程", real_contexts)
            
            # Should extract and map citations correctly
            assert len(result.citations) >= 2
            citation_indices = [c.index for c in result.citations]
            assert 1 in citation_indices
            assert 3 in citation_indices
            
            # Check citation mapping
            cite1 = next(c for c in result.citations if c.index == 1)
            assert cite1.chunk_id == "chunk_001"
            assert cite1.document_title == "企业级RAG实施指南"
    
    @pytest.mark.asyncio 
    async def test_structured_answer_generation(self, config, real_contexts):
        """Test generating structured answers."""
        config.extra_params["answer_format"] = "structured"
        generator = Generator(config)
        
        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "content": """
                # RAG实施步骤详解
                
                ## 1. 文档准备和预处理[3]
                - 格式转换：支持PDF、Word、HTML等格式
                - 数据清洗：去除无关信息
                - 标准化处理：统一格式和编码
                
                ## 2. 文档分块策略[3]
                - 考虑语义完整性
                - 平衡检索效率
                - 设置合适的chunk大小
                
                ## 3. 向量化和索引[3]
                - 选择嵌入模型
                - 构建向量索引
                - 优化检索性能
                
                ## 总结
                RAG系统的实施需要系统化的方法和careful planning[3]。
                """.strip(),
                "usage": {"total_tokens": 600}
            }
            
            query = "详细说明RAG系统的实施步骤"
            result = await generator.generate(query, real_contexts)
            
            assert "# RAG实施步骤详解" in result.answer
            assert "## 1. 文档准备和预处理" in result.answer
            assert "[3]" in result.answer
    
    @pytest.mark.asyncio
    async def test_multi_language_generation(self, config, real_contexts):
        """Test generation in different languages."""
        test_cases = [
            ("What is RAG?", "en", "Retrieval-Augmented Generation"),
            ("什么是RAG？", "zh", "检索增强生成"),
            ("RAGとは何ですか？", "ja", "検索拡張生成")
        ]
        
        generator = Generator(config)
        
        for query, lang, expected_term in test_cases:
            config.extra_params["language"] = lang
            
            with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
                # Simulate language-aware response
                if lang == "en":
                    content = f"RAG stands for {expected_term}[1], which combines retrieval and generation."
                elif lang == "zh":
                    content = f"RAG是{expected_term}[1]，它结合了检索和生成技术。"
                else:
                    content = f"RAGは{expected_term}[1]で、検索と生成を組み合わせた技術です。"
                
                mock_llm.return_value = {
                    "content": content,
                    "usage": {"total_tokens": 200}
                }
                
                result = await generator.generate(query, real_contexts[:1])
                assert expected_term in result.answer
    
    @pytest.mark.asyncio
    async def test_empty_context_handling(self, config):
        """Test handling queries with no relevant contexts."""
        generator = Generator(config)
        
        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "content": "抱歉，我在提供的文档中没有找到关于该问题的相关信息。请提供更多上下文或尝试其他问题。",
                "usage": {"total_tokens": 100}
            }
            
            result = await generator.generate("完全无关的问题", contexts=[])
            
            assert "没有找到" in result.answer or "抱歉" in result.answer
            assert len(result.citations) == 0
    
    @pytest.mark.asyncio
    async def test_token_limit_handling(self, config, real_contexts):
        """Test handling of token limits with many contexts."""
        # Create many contexts
        many_contexts = real_contexts * 10  # 30 contexts
        
        generator = Generator(config)
        generator.config.max_tokens = 500  # Set low limit
        
        with patch.object(generator, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "content": "基于部分相关文档的总结回答...",
                "usage": {"total_tokens": 490}
            }
            
            # Should truncate contexts to fit token limit
            result = await generator.generate("测试token限制", many_contexts)
            
            # Verify contexts were truncated
            call_args = mock_llm.call_args[0][0]  # Get the prompt
            assert len(call_args) < len(str(many_contexts))  # Prompt should be shorter
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self, config, real_contexts):
        """Test streaming generation mode."""
        config.extra_params["stream"] = True
        generator = Generator(config)
        
        async def mock_stream_llm(*args, **kwargs):
            """Mock streaming LLM responses."""
            chunks = [
                {"content": "RAG是", "usage": None},
                {"content": "一种先进的", "usage": None},
                {"content": "AI技术[1]。", "usage": None},
                {"content": "", "usage": {"total_tokens": 150}, "is_final": True}
            ]
            for chunk in chunks:
                yield chunk
        
        with patch.object(generator, '_stream_llm', mock_stream_llm):
            chunks_received = []
            async for chunk in generator.stream_generate("什么是RAG?", real_contexts[:1]):
                chunks_received.append(chunk)
            
            assert len(chunks_received) == 4
            assert chunks_received[0].content == "RAG是"
            assert chunks_received[-1].is_final
            assert chunks_received[-1].usage["total_tokens"] == 150
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, config, real_contexts):
        """Test error handling and recovery."""
        generator = Generator(config)
        
        call_count = 0
        
        async def flaky_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary API error")
            return {
                "content": "成功恢复后的答案",
                "usage": {"total_tokens": 100}
            }
        
        generator._call_llm = flaky_llm
        config.extra_params["max_retries"] = 3
        
        # Should retry and eventually succeed
        result = await generator.generate("测试重试", real_contexts[:1])
        
        assert call_count == 3
        assert "成功恢复" in result.answer


@pytest.mark.integration
class TestGeneratorWithProviders:
    """Test generator with different LLM providers."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DeepSeek API key not set")
    async def test_deepseek_real_call(self):
        """Test real DeepSeek API call."""
        config = RAGConfig(
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            llm_api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.1,
            max_tokens=200
        )
        
        generator = Generator(config)
        await generator.initialize()
        
        contexts = [
            RetrievalResult(
                chunk_id="test_1",
                content="Python是一种高级编程语言，以简洁清晰的语法著称。",
                score=0.9,
                metadata={"source": "python_guide"}
            )
        ]
        
        result = await generator.generate(
            query="用一句话介绍Python",
            contexts=contexts
        )
        
        assert result.answer
        assert len(result.answer) > 10
        assert result.usage["total_tokens"] > 0
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Fallback mechanism not yet implemented")
    async def test_provider_fallback(self):
        """Test fallback between providers."""
        config = RAGConfig(
            llm_provider="deepseek",
            llm_api_key="invalid-key",
            extra_params={
                "fallback_provider": "qwen",
                "fallback_api_key": os.getenv("DASHSCOPE_API_KEY", "test-key")
            }
        )
        
        generator = Generator(config)
        
        with patch('knowledge_core_engine.core.generation.generator.QwenProvider') as MockQwen:
            mock_qwen = AsyncMock()
            mock_qwen.generate.return_value = {
                "content": "Fallback response",
                "usage": {"total_tokens": 50}
            }
            MockQwen.return_value = mock_qwen
            
            # Should fallback to Qwen when DeepSeek fails
            contexts = []
            result = await generator.generate("test", contexts)
            
            assert "Fallback response" in result.answer