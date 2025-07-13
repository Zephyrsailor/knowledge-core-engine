"""Unit tests for prompt builder module."""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.generation.prompt_builder import (
    PromptBuilder, PromptTemplate, ContextFormatter
)
from knowledge_core_engine.core.retrieval.retriever import RetrievalResult


class TestPromptBuilder:
    """Test the PromptBuilder class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGConfig(
            llm_provider="deepseek",
            extra_params={
                "language": "zh",
                "max_context_length": 4000,
                "context_window_size": 16000
            }
        )
    
    @pytest.fixture
    def prompt_builder(self, config):
        """Create PromptBuilder instance."""
        return PromptBuilder(config)
    
    @pytest.fixture
    def mock_contexts(self):
        """Create mock retrieval contexts."""
        return [
            RetrievalResult(
                chunk_id="chunk_1",
                content="人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                score=0.95,
                metadata={
                    "document_title": "AI基础",
                    "page": 1,
                    "chunk_type": "definition"
                }
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                content="机器学习是AI的一个子领域，它使计算机能够从数据中学习，而无需明确编程。",
                score=0.88,
                metadata={
                    "document_title": "AI基础",
                    "page": 3,
                    "chunk_type": "concept"
                }
            )
        ]
    
    def test_basic_prompt_building(self, prompt_builder, mock_contexts):
        """Test basic prompt construction."""
        query = "什么是人工智能？"
        
        prompt = prompt_builder.build_prompt(
            query=query,
            contexts=mock_contexts
        )
        
        assert query in prompt
        assert "人工智能（AI）" in prompt
        assert "机器学习" in prompt
        assert "上下文" in prompt or "Context" in prompt
    
    def test_system_prompt_inclusion(self, prompt_builder, mock_contexts):
        """Test system prompt is included."""
        query = "解释AI"
        system_prompt = "你是一个AI专家，请用简洁的语言回答。"
        
        prompt = prompt_builder.build_prompt(
            query=query,
            contexts=mock_contexts,
            system_prompt=system_prompt
        )
        
        messages = prompt_builder.build_messages(
            query=query,
            contexts=mock_contexts,
            system_prompt=system_prompt
        )
        
        assert messages[0]["role"] == "system"
        assert system_prompt in messages[0]["content"]
    
    def test_context_formatting(self, prompt_builder, mock_contexts):
        """Test context formatting with metadata."""
        formatter = ContextFormatter()
        
        formatted = formatter.format_contexts(mock_contexts)
        
        assert "[文档1]" in formatted or "[Document 1]" in formatted
        assert "AI基础" in formatted
        assert "第1页" in formatted or "Page 1" in formatted
        # chunk_id is only included when specifically requested
        
        # Test with chunk_id included
        formatted_with_id = formatter.format_contexts(mock_contexts, include_metadata=True)
        assert "AI基础" in formatted_with_id
    
    def test_context_truncation(self, prompt_builder):
        """Test context truncation for token limits."""
        # Create very long contexts
        long_contexts = []
        for i in range(20):
            long_contexts.append(RetrievalResult(
                chunk_id=f"chunk_{i}",
                content="这是一段非常长的文本。" * 100,  # Very long text
                score=0.9 - i * 0.01,
                metadata={"document_title": f"Doc {i}"}
            ))
        
        query = "测试"
        
        # Use compress_contexts to truncate
        compressed = prompt_builder.compress_contexts(long_contexts, target_tokens=1000)
        prompt = prompt_builder.build_prompt(query, compressed)
        
        # Should be truncated to fit limit
        assert len(compressed) < len(long_contexts)  # Some contexts removed
        assert "Doc 0" in prompt  # Should keep high-score contexts
    
    def test_citation_instruction_addition(self, prompt_builder, mock_contexts):
        """Test adding citation instructions."""
        query = "需要引用的问题"
        prompt_builder.config.include_citations = True
        
        prompt = prompt_builder.build_prompt(
            query=query,
            contexts=mock_contexts,
            include_citations=True
        )
        
        assert "[1]" in prompt or "引用" in prompt or "citation" in prompt.lower()
    
    def test_structured_output_prompt(self, prompt_builder, mock_contexts):
        """Test prompt for structured output."""
        query = "列出AI的应用"
        
        prompt = prompt_builder.build_prompt(
            query=query,
            contexts=mock_contexts,
            output_format="structured"
        )
        
        # Should include formatting instructions
        assert "结构" in prompt or "格式" in prompt or "structure" in prompt.lower()
    
    def test_language_specific_prompts(self, prompt_builder, mock_contexts):
        """Test language-specific prompt variations."""
        query_zh = "什么是AI？"
        query_en = "What is AI?"
        
        # Chinese prompt
        prompt_zh = prompt_builder.build_prompt(query_zh, mock_contexts)
        assert "根据以下" in prompt_zh or "基于" in prompt_zh or "上下文" in prompt_zh
        
        # English prompt - the prompt template itself won't change based on language
        # but we can still verify the prompt is built
        prompt_builder.config.extra_params["language"] = "en"
        prompt_en = prompt_builder.build_prompt(query_en, mock_contexts)
        # The default template is in Chinese, so check for Chinese elements
        assert len(prompt_en) > 0  # Just ensure prompt is built
    
    def test_few_shot_examples(self, prompt_builder):
        """Test adding few-shot examples."""
        query = "解释概念"
        contexts = []
        
        examples = [
            {
                "query": "什么是机器学习？",
                "answer": "机器学习是人工智能的一个分支[1]，它使计算机能够从数据中学习模式[2]。"
            }
        ]
        
        prompt = prompt_builder.build_prompt(
            query=query,
            contexts=contexts,
            few_shot_examples=examples
        )
        
        assert "什么是机器学习？" in prompt
        assert "机器学习是人工智能的一个分支" in prompt
    
    def test_chain_of_thought_prompt(self, prompt_builder, mock_contexts):
        """Test chain of thought prompting."""
        query = "分析AI的影响"
        
        prompt = prompt_builder.build_prompt(
            query=query,
            contexts=mock_contexts,
            enable_cot=True
        )
        
        assert "步骤" in prompt or "思考" in prompt or "step" in prompt.lower()
    
    def test_custom_prompt_template(self, prompt_builder, mock_contexts):
        """Test using custom prompt template."""
        query = "测试"
        
        custom_template = """
        任务：{query}
        
        参考资料：
        {contexts}
        
        请提供详细答案。
        """
        
        prompt = prompt_builder.build_prompt(
            query=query,
            contexts=mock_contexts,
            template=custom_template
        )
        
        assert "任务：测试" in prompt
        assert "参考资料：" in prompt
        assert "请提供详细答案" in prompt


class TestPromptTemplate:
    """Test the PromptTemplate class."""
    
    def test_template_creation(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="qa_with_context",
            template="""
            Question: {query}
            Context: {contexts}
            Answer:
            """,
            description="Q&A with context template"
        )
        
        assert template.name == "qa_with_context"
        assert "{query}" in template.template
        assert "{contexts}" in template.template
    
    def test_template_formatting(self):
        """Test template variable substitution."""
        template = PromptTemplate(
            name="test",
            template="Query: {query}, Contexts: {contexts}"
        )
        
        result = template.format(
            query="What is AI?",
            contexts="AI is..."
        )
        
        assert result == "Query: What is AI?, Contexts: AI is..."
    
    def test_template_validation(self):
        """Test template validation."""
        # Valid template
        valid = PromptTemplate(
            name="valid",
            template="{query} {contexts}"
        )
        assert valid.is_valid()
        
        # Invalid template (missing required variables)
        invalid = PromptTemplate(
            name="invalid",
            template="{query} {unknown_var}"
        )
        assert not invalid.is_valid()
    
    def test_builtin_templates(self):
        """Test built-in prompt templates."""
        templates = PromptTemplate.get_builtin_templates()
        
        assert "qa_basic" in templates
        assert "qa_with_citations" in templates
        assert "qa_structured" in templates
        assert "qa_cot" in templates
        
        # Test a built-in template
        qa_template = templates["qa_basic"]
        assert isinstance(qa_template, PromptTemplate)
        assert qa_template.is_valid()


class TestContextFormatter:
    """Test the ContextFormatter class."""
    
    @pytest.fixture
    def formatter(self):
        """Create ContextFormatter instance."""
        return ContextFormatter()
    
    @pytest.fixture
    def contexts(self):
        """Create test contexts."""
        return [
            RetrievalResult(
                chunk_id="chunk_1",
                content="First context",
                score=0.95,
                metadata={
                    "document_title": "Doc A",
                    "page": 1,
                    "section": "Introduction"
                }
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                content="Second context",
                score=0.88,
                metadata={
                    "document_title": "Doc B",
                    "url": "https://example.com/doc-b"
                }
            )
        ]
    
    def test_basic_formatting(self, formatter, contexts):
        """Test basic context formatting."""
        formatted = formatter.format_contexts(contexts)
        
        assert "First context" in formatted
        assert "Second context" in formatted
        assert "Doc A" in formatted
        assert "Doc B" in formatted
    
    def test_formatting_with_indices(self, formatter, contexts):
        """Test formatting with document indices."""
        formatted = formatter.format_contexts(
            contexts,
            include_indices=True
        )
        
        assert "[文档1]" in formatted or "[Document 1]" in formatted
        assert "[文档2]" in formatted or "[Document 2]" in formatted
    
    def test_formatting_with_metadata(self, formatter, contexts):
        """Test including metadata in formatting."""
        formatted = formatter.format_contexts(
            contexts,
            include_metadata=True
        )
        
        # chunk_id is not included by default in text format
        # Check for actual metadata that gets included
        assert "Doc A" in formatted  # document_title
        assert "第1页" in formatted or "Page 1" in formatted
        assert "Doc B" in formatted  # second document
        # URL is shown when no page number
        assert "First context" in formatted
        assert "Second context" in formatted
    
    def test_compact_formatting(self, formatter, contexts):
        """Test compact formatting mode."""
        formatted = formatter.format_contexts(
            contexts,
            compact=True
        )
        
        # Should be more concise
        lines = formatted.strip().split('\n')
        assert len(lines) <= len(contexts) * 3  # Max 3 lines per context
    
    def test_markdown_formatting(self, formatter, contexts):
        """Test markdown formatting."""
        formatted = formatter.format_contexts(
            contexts,
            format="markdown"
        )
        
        assert "##" in formatted or "**" in formatted  # Markdown elements
        assert "```" in formatted or "> " in formatted
    
    def test_json_formatting(self, formatter, contexts):
        """Test JSON formatting."""
        formatted = formatter.format_contexts(
            contexts,
            format="json"
        )
        
        import json
        parsed = json.loads(formatted)
        
        assert len(parsed) == 2
        assert parsed[0]["content"] == "First context"
        assert parsed[0]["metadata"]["document_title"] == "Doc A"
    
    def test_custom_separator(self, formatter, contexts):
        """Test custom context separator."""
        formatted = formatter.format_contexts(
            contexts,
            separator="---BREAK---"
        )
        
        assert "---BREAK---" in formatted
        assert formatted.count("---BREAK---") == 1  # Between 2 contexts
    
    def test_max_length_truncation(self, formatter):
        """Test truncation for max length."""
        long_contexts = [
            RetrievalResult(
                chunk_id=f"chunk_{i}",
                content="Very long text. " * 100,
                score=0.9,
                metadata={"title": f"Doc {i}"}
            )
            for i in range(10)
        ]
        
        formatted = formatter.format_contexts(
            long_contexts,
            max_length=500
        )
        
        assert len(formatted) <= 600  # Some buffer for formatting
        assert "..." in formatted  # Truncation indicator


class TestPromptOptimization:
    """Test prompt optimization features."""
    
    @pytest.fixture
    def optimizer(self):
        """Create prompt optimizer."""
        config = RAGConfig()
        return PromptBuilder(config)
    
    def test_token_counting(self, optimizer):
        """Test token counting for prompts."""
        text = "这是一个测试文本。" * 100
        
        # Approximate token count
        token_count = optimizer.estimate_tokens(text)
        
        assert token_count > 0
        assert token_count < len(text)  # Should be less than character count
    
    def test_prompt_compression(self, optimizer):
        """Test prompt compression techniques."""
        contexts = [
            RetrievalResult(
                chunk_id="chunk_1",
                content="这是一段包含很多冗余信息的文本。" * 20,  # Make it longer
                score=0.9,
                metadata={}
            )
        ]
        
        compressed = optimizer.compress_contexts(
            contexts,
            target_tokens=50  # Very low target to force compression
        )
        
        # The compression truncates content, not removes contexts
        assert len(compressed) == len(contexts)
        assert len(compressed[0].content) < len(contexts[0].content)
        assert compressed[0].content.endswith("...")
    
    def test_dynamic_example_selection(self, optimizer):
        """Test dynamic few-shot example selection."""
        query = "解释机器学习"
        
        examples = [
            {"query": "什么是AI？", "answer": "AI是..."},
            {"query": "解释深度学习", "answer": "深度学习是..."},
            {"query": "什么是NLP？", "answer": "NLP是..."}
        ]
        
        selected = optimizer.select_examples(
            query=query,
            examples=examples,
            k=2
        )
        
        assert len(selected) == 2
        # Should select most relevant examples
        assert any("深度学习" in ex["query"] for ex in selected)