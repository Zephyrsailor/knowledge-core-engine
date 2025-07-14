"""Tests for query expansion functionality in the retriever."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.retrieval.retriever import Retriever


class TestQueryExpansion:
    """Test query expansion functionality."""
    
    @pytest.fixture
    def config_with_expansion(self):
        """Create config with query expansion enabled."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            enable_query_expansion=True,
            query_expansion_method="llm",
            query_expansion_count=3
        )
        return config
    
    @pytest.fixture
    def config_rule_based(self):
        """Create config with rule-based expansion."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            enable_query_expansion=True,
            query_expansion_method="rule_based",
            query_expansion_count=3
        )
        return config
    
    @pytest.mark.asyncio
    async def test_query_expansion_disabled(self):
        """Test that query expansion is not used when disabled."""
        config = RAGConfig(
            llm_provider="deepseek",
            embedding_provider="dashscope",
            enable_query_expansion=False
        )
        retriever = Retriever(config)
        
        # Mock the vector store to avoid initialization
        retriever._vector_store = AsyncMock()
        retriever._embedder = AsyncMock()
        retriever._initialized = True
        
        # Mock vector retrieve to check the query
        retriever._vector_retrieve = AsyncMock(return_value=[])
        
        await retriever.retrieve("什么是RAG技术？")
        
        # Should be called with original query
        retriever._vector_retrieve.assert_called_once_with("什么是RAG技术？", 10, None)
    
    @pytest.mark.asyncio
    async def test_llm_query_expansion(self, config_with_expansion):
        """Test LLM-based query expansion."""
        retriever = Retriever(config_with_expansion)
        
        # Mock LLM provider
        mock_llm_response = {
            "content": "RAG技术是什么\n检索增强生成\nRetrieval Augmented Generation"
        }
        
        with patch('knowledge_core_engine.core.retrieval.retriever.create_llm_provider') as mock_create:
            mock_provider = AsyncMock()
            mock_provider.generate = AsyncMock(return_value=mock_llm_response)
            mock_create.return_value = mock_provider
            
            # Test expansion
            expanded = await retriever._expand_query("什么是RAG技术？")
            
            # Should include original query
            assert "什么是RAG技术？" in expanded
            assert len(expanded) <= config_with_expansion.query_expansion_count
            
            # Check LLM was called with appropriate prompt
            mock_provider.generate.assert_called_once()
            call_args = mock_provider.generate.call_args
            messages = call_args[1]['messages']
            assert len(messages) == 1
            assert "什么是RAG技术？" in messages[0]['content']
    
    @pytest.mark.asyncio
    async def test_rule_based_expansion(self, config_rule_based):
        """Test rule-based query expansion."""
        retriever = Retriever(config_rule_based)
        
        # Test with known synonyms
        expanded = await retriever._expand_query("什么是RAG技术的优势？")
        
        assert "什么是RAG技术的优势？" in expanded  # Original
        # Should have variations with synonyms
        assert any("哪些" in q for q in expanded)  # 什么 -> 哪些
        assert any("优点" in q or "好处" in q for q in expanded)  # 优势 -> 优点/好处
    
    @pytest.mark.asyncio
    async def test_query_expansion_error_handling(self, config_with_expansion):
        """Test error handling in query expansion."""
        retriever = Retriever(config_with_expansion)
        
        # Mock LLM provider to raise error
        with patch('knowledge_core_engine.core.retrieval.retriever.create_llm_provider') as mock_create:
            mock_create.side_effect = Exception("LLM service unavailable")
            
            # Should not fail, just return original query
            expanded = await retriever._expand_query("test query")
            assert expanded == ["test query"]
    
    @pytest.mark.asyncio
    async def test_expanded_query_in_retrieval(self, config_with_expansion):
        """Test that expanded queries are used in retrieval."""
        retriever = Retriever(config_with_expansion)
        
        # Mock components
        retriever._initialized = True
        retriever._embedder = AsyncMock()
        retriever._vector_store = AsyncMock()
        
        # Mock expansion to return multiple queries
        retriever._expand_query = AsyncMock(
            return_value=["什么是RAG技术？", "RAG技术是什么", "检索增强生成"]
        )
        
        # Mock vector retrieve
        retriever._vector_retrieve = AsyncMock(return_value=[])
        
        await retriever.retrieve("什么是RAG技术？")
        
        # Should be called with combined expanded queries
        retriever._vector_retrieve.assert_called_once()
        call_args = retriever._vector_retrieve.call_args[0]
        combined_query = call_args[0]
        
        # Should contain all expanded queries
        assert "什么是RAG技术？" in combined_query
        assert "RAG技术是什么" in combined_query
        assert "检索增强生成" in combined_query
    
    @pytest.mark.asyncio
    async def test_rule_based_expansion_limits(self, config_rule_based):
        """Test that rule-based expansion respects count limits."""
        retriever = Retriever(config_rule_based)
        
        # Query with many possible expansions
        expanded = await retriever._expand_query("什么是技术的优势和问题")
        
        # Should not exceed configured count
        assert len(expanded) <= config_rule_based.query_expansion_count
        assert expanded[0] == "什么是技术的优势和问题"  # Original query first
    
    @pytest.mark.asyncio
    async def test_empty_query_expansion(self, config_with_expansion):
        """Test expansion with empty or whitespace query."""
        retriever = Retriever(config_with_expansion)
        
        # Empty query
        expanded = await retriever._expand_query("")
        assert expanded == [""]
        
        # Whitespace query
        expanded = await retriever._expand_query("   ")
        assert expanded == ["   "]
    
    @pytest.mark.asyncio
    async def test_llm_expansion_parsing(self, config_with_expansion):
        """Test parsing of LLM expansion response."""
        retriever = Retriever(config_with_expansion)
        
        # Test various response formats
        test_cases = [
            # Numbered list
            "1. query one\n2. query two\n3. query three",
            # Bullet points
            "- query one\n- query two\n- query three",
            # Mixed format
            "query one\n2. query two\n- query three\nquery four",
            # With empty lines
            "query one\n\nquery two\n\nquery three"
        ]
        
        with patch('knowledge_core_engine.core.retrieval.retriever.create_llm_provider') as mock_create:
            mock_provider = AsyncMock()
            mock_create.return_value = mock_provider
            
            for response_content in test_cases:
                mock_provider.generate = AsyncMock(
                    return_value={"content": response_content}
                )
                
                expanded = await retriever._expand_query("test")
                
                # Should have multiple queries
                assert len(expanded) > 1
                assert expanded[0] == "test"  # Original first
                # Should not include numbered prefixes
                assert not any(q.strip().startswith(('1.', '2.', '-')) for q in expanded[1:])
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_with_expansion(self, config_with_expansion):
        """Test query expansion works with hybrid retrieval."""
        config_with_expansion.retrieval_strategy = "hybrid"
        retriever = Retriever(config_with_expansion)
        
        # Mock components
        retriever._initialized = True
        retriever._expand_query = AsyncMock(
            return_value=["original", "expanded1", "expanded2"]
        )
        retriever._vector_retrieve = AsyncMock(return_value=[])
        retriever._bm25_retrieve = AsyncMock(return_value=[])
        
        await retriever.retrieve("original")
        
        # Both retrieval methods should use expanded query
        combined_query = "original expanded1 expanded2"
        retriever._vector_retrieve.assert_called_with(combined_query, 20, None)
        retriever._bm25_retrieve.assert_called_with(combined_query, 20, None)