"""Integration tests for KnowledgeEngine with enhanced features."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from knowledge_core_engine import KnowledgeEngine


class TestKnowledgeEngineEnhancedFeatures:
    """Test KnowledgeEngine with enhanced configuration features."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_documents(self, temp_dir):
        """Create test documents."""
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        
        # Create test markdown file
        (docs_dir / "test.md").write_text("""
# RAG技术介绍

## 什么是RAG

RAG（Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的AI技术。

## RAG的优势

1. 提高准确性
2. 减少幻觉
3. 知识可更新

## 实现方法

RAG通过向量数据库存储知识，并在生成时检索相关内容。
""")
        
        return docs_dir
    
    @pytest.mark.asyncio
    async def test_engine_with_hierarchical_chunking(self, temp_dir, test_documents):
        """Test engine with hierarchical chunking enabled."""
        engine = KnowledgeEngine(
            persist_directory=str(Path(temp_dir) / "knowledge_base"),
            enable_hierarchical_chunking=True,
            enable_semantic_chunking=False,
            chunk_size=200,
            chunk_overlap=50
        )
        
        # Mock LLM and embedding providers
        with patch('knowledge_core_engine.core.generation.generator.create_llm_provider') as mock_llm, \
             patch('knowledge_core_engine.core.embedding.embedder.create_embedding_provider') as mock_embed:
            
            # Setup mocks
            mock_llm_provider = AsyncMock()
            mock_llm_provider.generate = AsyncMock(return_value={
                "content": "RAG是检索增强生成技术。"
            })
            mock_llm.return_value = mock_llm_provider
            
            mock_embed_provider = AsyncMock()
            mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1536)
            mock_embed.return_value = mock_embed_provider
            
            # Add documents
            result = await engine.add(test_documents)
            assert result["processed_files"] == 1
            assert result["total_chunks"] > 0
            
            # Verify hierarchical chunking was used
            assert engine.config.enable_hierarchical_chunking is True
            assert engine._chunker.chunker.__class__.__name__ == "EnhancedChunker"
    
    @pytest.mark.asyncio
    async def test_engine_with_metadata_enhancement(self, temp_dir, test_documents):
        """Test engine with metadata enhancement enabled."""
        engine = KnowledgeEngine(
            persist_directory=str(Path(temp_dir) / "knowledge_base"),
            enable_metadata_enhancement=True
        )
        
        # Mock providers
        with patch('knowledge_core_engine.core.generation.generator.create_llm_provider') as mock_llm, \
             patch('knowledge_core_engine.core.embedding.embedder.create_embedding_provider') as mock_embed, \
             patch.object(engine._metadata_enhancer, 'enhance_chunk') as mock_enhance:
            
            # Setup mocks
            mock_llm_provider = AsyncMock()
            mock_llm.return_value = mock_llm_provider
            
            mock_embed_provider = AsyncMock()
            mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1536)
            mock_embed.return_value = mock_embed_provider
            
            # Mock metadata enhancement
            async def enhance_side_effect(chunk):
                chunk.metadata.update({
                    "summary": "Test summary",
                    "questions": ["Q1", "Q2"],
                    "keywords": ["RAG", "AI"]
                })
                return chunk
            
            mock_enhance.side_effect = enhance_side_effect
            
            # Add documents
            result = await engine.add(test_documents)
            
            # Verify metadata enhancer was called
            assert mock_enhance.called
            assert engine._metadata_enhancer is not None
    
    @pytest.mark.asyncio
    async def test_engine_with_hybrid_retrieval(self, temp_dir):
        """Test engine with hybrid retrieval strategy."""
        engine = KnowledgeEngine(
            persist_directory=str(Path(temp_dir) / "knowledge_base"),
            retrieval_strategy="hybrid",
            vector_weight=0.6,
            bm25_weight=0.4,
            fusion_method="weighted"
        )
        
        # Verify configuration
        assert engine.config.retrieval_strategy == "hybrid"
        assert engine.config.vector_weight == 0.6
        assert engine.config.bm25_weight == 0.4
        assert engine.config.fusion_method == "weighted"
    
    @pytest.mark.asyncio
    async def test_engine_with_query_expansion(self, temp_dir):
        """Test engine with query expansion enabled."""
        engine = KnowledgeEngine(
            persist_directory=str(Path(temp_dir) / "knowledge_base"),
            enable_query_expansion=True,
            query_expansion_method="rule_based",
            query_expansion_count=3
        )
        
        # Mock components
        with patch('knowledge_core_engine.core.generation.generator.create_llm_provider') as mock_llm, \
             patch('knowledge_core_engine.core.embedding.embedder.create_embedding_provider') as mock_embed:
            
            mock_llm_provider = AsyncMock()
            mock_llm_provider.generate = AsyncMock(return_value={"content": "测试答案"})
            mock_llm.return_value = mock_llm_provider
            
            mock_embed_provider = AsyncMock()
            mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1536)
            mock_embed.return_value = mock_embed_provider
            
            # Mock retriever's expand_query method
            await engine._ensure_initialized()
            original_expand = engine._retriever._expand_query
            engine._retriever._expand_query = AsyncMock(
                return_value=["什么是RAG", "RAG是什么", "RAG技术"]
            )
            
            # Ask a question
            answer = await engine.ask("什么是RAG")
            
            # Verify query expansion was called
            engine._retriever._expand_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_engine_with_reranking(self, temp_dir):
        """Test engine with reranking enabled."""
        engine = KnowledgeEngine(
            persist_directory=str(Path(temp_dir) / "knowledge_base"),
            enable_reranking=True,
            reranker_model="bge-reranker-v2-m3",
            rerank_top_k=3
        )
        
        # Verify configuration and components
        assert engine.config.enable_reranking is True
        assert engine.config.reranker_model == "bge-reranker-v2-m3"
        assert engine.config.rerank_top_k == 3
        
        await engine._ensure_initialized()
        assert engine._reranker is not None
    
    @pytest.mark.asyncio
    async def test_engine_comprehensive_features(self, temp_dir, test_documents):
        """Test engine with all enhanced features enabled."""
        engine = KnowledgeEngine(
            persist_directory=str(Path(temp_dir) / "knowledge_base"),
            # Chunking features
            enable_hierarchical_chunking=True,
            enable_metadata_enhancement=True,
            chunk_size=512,
            chunk_overlap=64,
            # Retrieval features
            retrieval_strategy="hybrid",
            enable_query_expansion=True,
            query_expansion_method="rule_based",
            # Reranking features
            enable_reranking=True,
            reranker_model="bge-reranker-v2-m3",
            rerank_top_k=5
        )
        
        # Verify all configurations
        assert engine.config.enable_hierarchical_chunking is True
        assert engine.config.enable_metadata_enhancement is True
        assert engine.config.retrieval_strategy == "hybrid"
        assert engine.config.enable_query_expansion is True
        assert engine.config.enable_reranking is True
        
        # Mock all external dependencies
        with patch('knowledge_core_engine.core.generation.generator.create_llm_provider') as mock_llm, \
             patch('knowledge_core_engine.core.embedding.embedder.create_embedding_provider') as mock_embed:
            
            # Setup basic mocks
            mock_llm_provider = AsyncMock()
            mock_llm_provider.generate = AsyncMock(return_value={
                "content": "RAG是一种先进的AI技术，结合了检索和生成。[1]"
            })
            mock_llm.return_value = mock_llm_provider
            
            mock_embed_provider = AsyncMock()
            mock_embed_provider.embed = AsyncMock(return_value=[0.1] * 1536)
            mock_embed.return_value = mock_embed_provider
            
            await engine._ensure_initialized()
            
            # Mock metadata enhancer
            if engine._metadata_enhancer:
                engine._metadata_enhancer.enhance_chunk = AsyncMock(
                    side_effect=lambda chunk: chunk
                )
            
            # Mock reranker
            if engine._reranker:
                engine._reranker.rerank = AsyncMock(
                    side_effect=lambda q, results, top_k: results[:top_k]
                )
            
            # Add documents
            result = await engine.add(test_documents)
            assert result["processed_files"] == 1
            
            # Ask a question
            answer = await engine.ask("什么是RAG技术？", return_details=True)
            
            assert "answer" in answer
            assert "contexts" in answer
            assert "citations" in answer
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, temp_dir):
        """Test that engine works with minimal/default configuration."""
        # Should work with just the basic parameters
        engine = KnowledgeEngine(
            persist_directory=str(Path(temp_dir) / "knowledge_base")
        )
        
        # Verify defaults are applied
        assert engine.config.enable_hierarchical_chunking is False
        assert engine.config.enable_semantic_chunking is True
        assert engine.config.retrieval_strategy == "hybrid"
        assert engine.config.enable_query_expansion is False
        assert engine.config.enable_reranking is False
        
        # Should initialize without errors
        await engine._ensure_initialized()
        assert engine._initialized is True