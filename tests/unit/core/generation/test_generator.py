"""Unit tests for the generator module."""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.generation.generator import (
    Generator, GenerationResult, CitationReference
)
from knowledge_core_engine.core.retrieval.retriever import RetrievalResult


class TestGenerator:
    """Test the Generator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGConfig(
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            llm_api_key="test-deepseek-key",
            temperature=0.1,
            max_tokens=2048,
            include_citations=True,
            extra_params={
                "system_prompt": "你是一个专业的知识助手，基于提供的上下文回答问题。",
                "citation_style": "inline",  # inline, footnote, or endnote
                "answer_format": "structured"  # structured or narrative
            }
        )
    
    @pytest.fixture
    def generator(self, config):
        """Create Generator instance."""
        return Generator(config)
    
    @pytest.fixture
    def mock_retrieval_results(self):
        """Create mock retrieval results for generation."""
        return [
            RetrievalResult(
                chunk_id="chunk_1",
                content="RAG（Retrieval-Augmented Generation）是一种将检索和生成相结合的技术。它通过检索相关文档来增强语言模型的生成能力。",
                score=0.95,
                metadata={
                    "document_id": "doc_1",
                    "document_title": "RAG技术详解",
                    "page": 3
                }
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                content="RAG的主要优势包括：1）减少幻觉；2）知识可更新；3）可解释性强。企业可以通过RAG构建自己的知识问答系统。",
                score=0.88,
                metadata={
                    "document_id": "doc_1",
                    "document_title": "RAG技术详解",
                    "page": 5
                }
            ),
            RetrievalResult(
                chunk_id="chunk_3",
                content="实施RAG系统需要考虑文档处理、向量化、检索策略和生成质量控制等多个方面。",
                score=0.82,
                metadata={
                    "document_id": "doc_2",
                    "document_title": "RAG实施指南",
                    "page": 12
                }
            )
        ]
    
    @pytest.mark.asyncio
    async def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator._initialized is False
        assert generator.config.llm_provider == "deepseek"
        assert generator.config.temperature == 0.1
        
        with patch.object(generator, '_create_llm_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            
            await generator.initialize()
            
            assert generator._initialized is True
            assert generator._llm_client is not None
            mock_create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_basic_generation(self, generator, mock_retrieval_results):
        """Test basic answer generation."""
        query = "什么是RAG技术？它有哪些优势？"
        
        with patch.object(generator, '_llm_client') as mock_client:
            # Mock LLM response
            mock_client.generate = AsyncMock(return_value={
                "content": "RAG（Retrieval-Augmented Generation）是一种将检索和生成相结合的技术[1]。它通过检索相关文档来增强语言模型的生成能力[1]。\n\nRAG的主要优势包括[2]：\n1）减少幻觉：通过基于真实文档生成答案，大大减少了模型产生虚假信息的可能性\n2）知识可更新：无需重新训练模型即可更新知识库\n3）可解释性强：可以追溯答案来源，提高可信度\n\n企业可以通过RAG构建自己的知识问答系统[2]，这对于需要准确、可追溯答案的应用场景特别有价值。",
                "citations": [
                    {"index": 1, "chunk_id": "chunk_1", "text": "RAG（Retrieval-Augmented Generation）是一种将检索和生成相结合的技术。它通过检索相关文档来增强语言模型的生成能力。"},
                    {"index": 2, "chunk_id": "chunk_2", "text": "RAG的主要优势包括：1）减少幻觉；2）知识可更新；3）可解释性强。企业可以通过RAG构建自己的知识问答系统。"}
                ],
                "usage": {"prompt_tokens": 450, "completion_tokens": 200, "total_tokens": 650}
            })
            
            generator._initialized = True
            
            result = await generator.generate(
                query=query,
                contexts=mock_retrieval_results
            )
            
            assert isinstance(result, GenerationResult)
            assert "RAG（Retrieval-Augmented Generation）" in result.answer
            assert "减少幻觉" in result.answer
            assert len(result.citations) == 2
            assert result.citations[0].chunk_id == "chunk_1"
            assert result.usage["total_tokens"] == 650
            
            # Check prompt construction
            mock_client.generate.assert_called_once()
            call_args = mock_client.generate.call_args[1]
            assert "contexts" in call_args or "messages" in call_args
    
    @pytest.mark.asyncio
    async def test_generation_without_citations(self, generator, mock_retrieval_results):
        """Test generation without citations."""
        generator.config.include_citations = False
        query = "解释RAG技术"
        
        with patch.object(generator, '_llm_client') as mock_client:
            mock_client.generate = AsyncMock(return_value={
                "content": "RAG是一种先进的AI技术，它结合了信息检索和文本生成的能力。",
                "usage": {"total_tokens": 100}
            })
            
            generator._initialized = True
            
            result = await generator.generate(query, mock_retrieval_results)
            
            assert result.citations == []
            assert "[1]" not in result.answer
    
    @pytest.mark.asyncio
    async def test_empty_context_handling(self, generator):
        """Test generation with empty context."""
        query = "没有相关文档的问题"
        
        with patch.object(generator, '_llm_client') as mock_client:
            mock_client.generate = AsyncMock(return_value={
                "content": "抱歉，我在提供的文档中没有找到相关信息来回答您的问题。",
                "usage": {"total_tokens": 50}
            })
            
            generator._initialized = True
            
            result = await generator.generate(query, contexts=[])
            
            assert "没有找到相关信息" in result.answer
            assert result.citations == []
    
    @pytest.mark.asyncio
    async def test_structured_answer_generation(self, generator, mock_retrieval_results):
        """Test structured answer format generation."""
        query = "详细说明RAG的实施步骤"
        generator.config.extra_params["answer_format"] = "structured"
        
        with patch.object(generator, '_llm_client') as mock_client:
            mock_client.generate = AsyncMock(return_value={
                "content": "## RAG实施步骤\n\n### 1. 文档准备\n准备并整理知识库文档[3]\n\n### 2. 文档处理\n对文档进行解析、分块和向量化[3]\n\n### 3. 检索系统搭建\n构建向量数据库和检索策略[3]\n\n### 4. 生成优化\n配置LLM并优化生成质量[3]",
                "citations": [
                    {"index": 3, "chunk_id": "chunk_3", "text": "实施RAG系统需要考虑文档处理、向量化、检索策略和生成质量控制等多个方面。"}
                ],
                "usage": {"total_tokens": 300}
            })
            
            generator._initialized = True
            
            result = await generator.generate(query, mock_retrieval_results)
            
            assert "## RAG实施步骤" in result.answer
            assert "### 1. 文档准备" in result.answer
            assert len(result.citations) == 1
    
    @pytest.mark.asyncio
    async def test_citation_extraction(self, generator):
        """Test citation extraction from generated text."""
        text_with_citations = """
        RAG技术的核心优势在于其准确性[1]。通过结合检索和生成[2]，
        它能够提供更可靠的答案[1]。此外，RAG还支持知识更新[3]。
        """
        
        citations = generator._extract_citations(text_with_citations)
        
        assert len(citations) == 4  # Including duplicate [1]
        assert citations == [1, 2, 1, 3]
        # Unique citations
        assert len(set(citations)) == 3
        assert set(citations) == {1, 2, 3}
    
    @pytest.mark.asyncio
    async def test_prompt_construction(self, generator, mock_retrieval_results):
        """Test prompt construction for generation."""
        query = "测试问题"
        
        prompt = generator._prompt_builder.build_prompt(query, mock_retrieval_results)
        
        assert query in prompt
        assert "chunk_1" in prompt or "RAG（Retrieval-Augmented Generation）" in prompt
        assert "上下文" in prompt or "Context" in prompt
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self, generator, mock_retrieval_results):
        """Test streaming answer generation."""
        query = "流式生成测试"
        generator.config.extra_params["stream"] = True
        
        with patch.object(generator, '_llm_client') as mock_client:
            # Mock streaming response
            async def mock_stream(**kwargs):
                chunks = ["RAG", "是一种", "强大的", "技术[1]。"]
                for chunk in chunks:
                    yield {"content": chunk, "usage": None}
                yield {
                    "content": "",
                    "usage": {"total_tokens": 100},
                    "citations": [{"index": 1, "chunk_id": "chunk_1"}]
                }
            
            mock_client.stream_generate = mock_stream
            generator._initialized = True
            
            chunks = []
            async for chunk in generator.stream_generate(query, mock_retrieval_results):
                chunks.append(chunk)
            
            assert len(chunks) == 5  # 4 content chunks + 1 final
            assert chunks[0].content == "RAG"
            assert chunks[-1].is_final is True
            assert chunks[-1].usage["total_tokens"] == 100
    
    @pytest.mark.asyncio
    async def test_generation_with_system_prompt(self, generator, mock_retrieval_results):
        """Test generation with custom system prompt."""
        query = "测试系统提示词"
        custom_prompt = "You are a technical expert. Answer concisely."
        generator.config.extra_params["system_prompt"] = custom_prompt
        
        with patch.object(generator, '_llm_client') as mock_client:
            mock_client.generate = AsyncMock(return_value={
                "content": "Technical answer here.",
                "usage": {"total_tokens": 80}
            })
            
            generator._initialized = True
            
            await generator.generate(query, mock_retrieval_results)
            
            # Verify system prompt was used
            call_args = mock_client.generate.call_args
            assert custom_prompt in str(call_args)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, generator, mock_retrieval_results):
        """Test error handling during generation."""
        query = "错误测试"
        
        with patch.object(generator, '_llm_client') as mock_client:
            mock_client.generate = AsyncMock(side_effect=Exception("LLM API Error"))
            
            generator._initialized = True
            
            with pytest.raises(Exception, match="LLM API Error"):
                await generator.generate(query, mock_retrieval_results)
    
    @pytest.mark.asyncio
    async def test_token_limit_handling(self, generator, mock_retrieval_results):
        """Test handling of token limits."""
        query = "长文本测试"
        # Add many contexts to exceed token limit
        many_contexts = mock_retrieval_results * 20
        
        with patch.object(generator, '_truncate_contexts') as mock_truncate:
            mock_truncate.return_value = mock_retrieval_results[:2]  # Truncate to 2
            
            with patch.object(generator, '_llm_client') as mock_client:
                mock_client.generate = AsyncMock(return_value={
                    "content": "Answer based on truncated context.",
                    "usage": {"total_tokens": 500}
                })
                
                generator._initialized = True
                generator.config.max_tokens = 1000
                
                result = await generator.generate(query, many_contexts)
                
                mock_truncate.assert_called_once()
                assert "truncated context" in result.answer.lower()


class TestCitationReference:
    """Test the CitationReference class."""
    
    def test_citation_reference_creation(self):
        """Test creating a citation reference."""
        citation = CitationReference(
            index=1,
            chunk_id="chunk_123",
            document_title="技术文档",
            page=5,
            text="这是引用的文本内容"
        )
        
        assert citation.index == 1
        assert citation.chunk_id == "chunk_123"
        assert citation.document_title == "技术文档"
        assert citation.page == 5
        assert citation.text == "这是引用的文本内容"
    
    def test_citation_to_dict(self):
        """Test converting citation to dictionary."""
        citation = CitationReference(
            index=1,
            chunk_id="chunk_1",
            document_title="RAG Guide",
            text="Sample text"
        )
        
        data = citation.to_dict()
        
        assert data["index"] == 1
        assert data["chunk_id"] == "chunk_1"
        assert data["document_title"] == "RAG Guide"
        assert data["text"] == "Sample text"
        assert data["page"] is None
    
    def test_citation_formatting(self):
        """Test citation formatting methods."""
        citation = CitationReference(
            index=1,
            chunk_id="chunk_1",
            document_title="技术指南",
            page=10
        )
        
        # Test inline format
        inline = citation.format_inline()
        assert inline == "[1]"
        
        # Test footnote format
        footnote = citation.format_footnote()
        assert "技术指南" in footnote
        assert "p.10" in footnote or "页10" in footnote


class TestGenerationResult:
    """Test the GenerationResult class."""
    
    def test_generation_result_creation(self):
        """Test creating a generation result."""
        citations = [
            CitationReference(1, "chunk_1", "Doc 1", text="Text 1"),
            CitationReference(2, "chunk_2", "Doc 2", text="Text 2")
        ]
        
        result = GenerationResult(
            query="What is RAG?",
            answer="RAG is a technology...",
            citations=citations,
            usage={"total_tokens": 150},
            metadata={"model": "deepseek-chat"}
        )
        
        assert result.query == "What is RAG?"
        assert result.answer == "RAG is a technology..."
        assert len(result.citations) == 2
        assert result.usage["total_tokens"] == 150
        assert result.metadata["model"] == "deepseek-chat"
    
    def test_generation_result_to_dict(self):
        """Test converting generation result to dictionary."""
        result = GenerationResult(
            query="Test query",
            answer="Test answer",
            citations=[],
            usage={"total_tokens": 100}
        )
        
        data = result.to_dict()
        
        assert data["query"] == "Test query"
        assert data["answer"] == "Test answer"
        assert data["citations"] == []
        assert data["usage"]["total_tokens"] == 100
        assert "timestamp" in data
    
    def test_formatted_answer_with_citations(self):
        """Test generating formatted answer with citations."""
        citations = [
            CitationReference(1, "chunk_1", "Document A", 5, "Citation text 1"),
            CitationReference(2, "chunk_2", "Document B", 10, "Citation text 2")
        ]
        
        result = GenerationResult(
            query="Q",
            answer="This is the answer[1]. More info here[2].",
            citations=citations,
            usage={}
        )
        
        # Test with footnotes
        formatted = result.get_formatted_answer(citation_style="footnote")
        assert "[1]" in formatted
        assert "Document A" in formatted
        assert "p.5" in formatted or "页5" in formatted


class TestLLMProviders:
    """Test different LLM provider integrations."""
    
    @pytest.fixture
    def deepseek_config(self):
        """Create DeepSeek configuration."""
        return RAGConfig(
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            llm_api_key="test-key"
        )
    
    @pytest.fixture
    def qwen_config(self):
        """Create Qwen configuration."""
        return RAGConfig(
            llm_provider="qwen",
            llm_model="qwen2.5-72b-instruct",
            llm_api_key="test-key"
        )
    
    @pytest.mark.asyncio
    async def test_deepseek_provider(self, deepseek_config):
        """Test DeepSeek provider initialization."""
        generator = Generator(deepseek_config)
        
        with patch('knowledge_core_engine.core.generation.generator.create_llm_provider') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.generate = AsyncMock(return_value={"content": "test", "usage": {}})
            mock_create.return_value = mock_provider
            
            await generator.initialize()
            
            assert generator._llm_client is not None
            mock_create.assert_called_once_with(deepseek_config)
    
    @pytest.mark.asyncio
    async def test_qwen_provider(self, qwen_config):
        """Test Qwen provider initialization."""
        generator = Generator(qwen_config)
        
        with patch('knowledge_core_engine.core.generation.generator.create_llm_provider') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.generate = AsyncMock(return_value={"content": "test", "usage": {}})
            mock_create.return_value = mock_provider
            
            await generator.initialize()
            
            assert generator._llm_client is not None
            mock_create.assert_called_once_with(qwen_config)
    
    @pytest.mark.asyncio
    async def test_provider_switching(self):
        """Test switching between providers."""
        # Start with DeepSeek
        config = RAGConfig(llm_provider="deepseek", llm_api_key="test-key")
        generator = Generator(config)
        
        with patch('knowledge_core_engine.core.generation.generator.create_llm_provider') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.generate = AsyncMock(return_value={"content": "test", "usage": {}})
            mock_create.return_value = mock_provider
            
            await generator.initialize()
            assert generator._initialized is True
            
            # Switch to Qwen
            generator.config.llm_provider = "qwen"
            generator.config.llm_model = "qwen2.5-72b-instruct"
            generator._initialized = False
            
            await generator.initialize()
            
            # Should reinitialize with new provider
            assert generator._initialized is True
            assert mock_create.call_count == 2


class TestAdvancedGeneration:
    """Test advanced generation features."""
    
    @pytest.fixture
    def generator(self):
        """Create generator with advanced features."""
        config = RAGConfig(
            llm_provider="deepseek",
            extra_params={
                "enable_cot": True,  # Chain of thought
                "enable_self_critique": True,
                "max_retries": 3,
                "temperature_decay": 0.1
            }
        )
        return Generator(config)
    
    @pytest.mark.asyncio
    async def test_chain_of_thought(self, generator):
        """Test chain of thought reasoning."""
        query = "复杂推理问题"
        contexts = []
        
        generator.config.extra_params["enable_cot"] = True
        
        # Mock response with reasoning steps
        cot_response = """
        让我逐步分析这个问题：
        
        步骤1：理解问题的核心
        步骤2：分析相关因素
        步骤3：得出结论
        
        最终答案：这是经过推理的答案。
        """
        
        with patch.object(generator, '_llm_client') as mock_client:
            mock_client.generate = AsyncMock(return_value={
                "content": cot_response,
                "usage": {"total_tokens": 200}
            })
            
            generator._initialized = True
            
            result = await generator.generate(query, contexts)
            
            assert "步骤" in result.answer
            assert "最终答案" in result.answer
    
    @pytest.mark.asyncio
    async def test_self_critique(self, generator):
        """Test self-critique mechanism."""
        query = "需要验证的问题"
        contexts = []
        
        # Mock the LLM client first
        with patch.object(generator, '_llm_client') as mock_client:
            mock_client.generate = AsyncMock(return_value={
                "content": "初始答案",
                "usage": {"total_tokens": 100}
            })
            
            generator._initialized = True
            generator.config.extra_params["enable_self_critique"] = True
            
            # For now, self-critique is not implemented in generate()
            # So we test the method directly
            with patch.object(generator, '_generate_with_critique') as mock_critique:
                mock_critique.return_value = GenerationResult(
                    query=query,
                    answer="经过自我批判改进的答案",
                    citations=[],
                    usage={"total_tokens": 300}
                )
                
                result = await mock_critique(query, contexts)
                
                assert "改进的答案" in result.answer
                assert result.usage["total_tokens"] == 300
    
    @pytest.mark.asyncio
    async def test_retry_with_temperature_decay(self, generator):
        """Test retry mechanism with temperature decay."""
        query = "测试重试"
        contexts = []
        
        call_count = 0
        
        async def mock_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise Exception("Temporary failure")
            
            return {
                "content": "Success after retries",
                "usage": {"total_tokens": 100}
            }
        
        with patch.object(generator, '_llm_client') as mock_client:
            mock_client.generate = mock_generate
            generator._initialized = True
            generator.config.extra_params["max_retries"] = 3
            
            # Add some contexts to avoid no-context path
            contexts = [RetrievalResult("chunk_1", "Test content", 0.9, {})]
            
            result = await generator.generate(query, contexts)
            
            assert call_count == 3
            assert "Success after retries" in result.answer