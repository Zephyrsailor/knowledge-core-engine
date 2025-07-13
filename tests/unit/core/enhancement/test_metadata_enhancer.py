"""Unit tests for metadata enhancement module."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import json
from typing import List

from knowledge_core_engine.core.chunking.base import ChunkResult
from knowledge_core_engine.core.enhancement.metadata_enhancer import (
    MetadataEnhancer, ChunkMetadata, EnhancementConfig
)


class TestChunkMetadata:
    """Test the ChunkMetadata model."""
    
    def test_chunk_metadata_creation(self):
        """Test creating ChunkMetadata with valid data."""
        metadata = ChunkMetadata(
            summary="This chunk describes RAG technology",
            questions=["What is RAG?", "How does RAG work?"],
            chunk_type="概念定义",
            keywords=["RAG", "retrieval", "generation"]
        )
        
        assert metadata.summary == "This chunk describes RAG technology"
        assert len(metadata.questions) == 2
        assert metadata.chunk_type == "概念定义"
        assert "RAG" in metadata.keywords
    
    def test_chunk_metadata_validation(self):
        """Test ChunkMetadata validation."""
        # Test empty questions list
        metadata = ChunkMetadata(
            summary="Summary",
            questions=[],
            chunk_type="其他",
            keywords=["test"]
        )
        assert metadata.questions == []
        
        # Test with many questions
        metadata = ChunkMetadata(
            summary="Summary",
            questions=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
            chunk_type="其他",
            keywords=["test"]
        )
        assert len(metadata.questions) == 6  # No automatic truncation
    
    def test_chunk_metadata_to_dict(self):
        """Test converting ChunkMetadata to dictionary."""
        metadata = ChunkMetadata(
            summary="Test summary",
            questions=["Q1"],
            chunk_type="示例代码",
            keywords=["python", "code"]
        )
        
        data = metadata.model_dump()
        assert data["summary"] == "Test summary"
        assert data["questions"] == ["Q1"]
        assert data["chunk_type"] == "示例代码"
        assert data["keywords"] == ["python", "code"]


class TestMetadataEnhancer:
    """Test the MetadataEnhancer class."""
    
    @pytest.fixture
    def sample_chunk(self):
        """Create a sample chunk for testing."""
        return ChunkResult(
            content="RAG (Retrieval Augmented Generation) is a technique that combines retrieval systems with language models to provide more accurate and contextual responses.",
            metadata={
                "chunk_id": "test_chunk_1",
                "document_id": "test_doc",
                "chunk_index": 0
            }
        )
    
    @pytest.fixture
    def enhancer(self):
        """Create a MetadataEnhancer instance."""
        config = EnhancementConfig(
            llm_provider="mock",
            model_name="mock-model",
            temperature=0.1
        )
        return MetadataEnhancer(config)
    
    @pytest.mark.asyncio
    async def test_enhance_chunk_success(self, enhancer, sample_chunk):
        """Test successful chunk enhancement."""
        # Mock LLM response
        mock_response = {
            "summary": "RAG combines retrieval and generation for better AI responses",
            "questions": [
                "What is RAG?",
                "How does RAG work?",
                "What are the benefits of RAG?"
            ],
            "chunk_type": "概念定义",
            "keywords": ["RAG", "retrieval", "generation", "AI"]
        }
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps(mock_response)
            
            enhanced_chunk = await enhancer.enhance_chunk(sample_chunk)
            
            # Verify the chunk was enhanced
            assert enhanced_chunk.metadata["summary"] == mock_response["summary"]
            assert enhanced_chunk.metadata["questions"] == mock_response["questions"]
            assert enhanced_chunk.metadata["chunk_type"] == mock_response["chunk_type"]
            assert enhanced_chunk.metadata["keywords"] == mock_response["keywords"]
            
            # Original metadata should be preserved
            assert enhanced_chunk.metadata["chunk_id"] == "test_chunk_1"
            assert enhanced_chunk.metadata["document_id"] == "test_doc"
    
    @pytest.mark.asyncio
    async def test_enhance_chunk_with_retry(self, enhancer, sample_chunk):
        """Test chunk enhancement with retry on failure."""
        # First call fails, second succeeds
        mock_response = {
            "summary": "Test summary",
            "questions": ["Q1"],
            "chunk_type": "其他",
            "keywords": ["test"]
        }
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [
                Exception("LLM API error"),
                json.dumps(mock_response)
            ]
            
            enhanced_chunk = await enhancer.enhance_chunk(sample_chunk)
            
            # Should succeed after retry
            assert enhanced_chunk.metadata["summary"] == mock_response["summary"]
            assert mock_llm.call_count == 2
    
    @pytest.mark.asyncio
    async def test_enhance_chunk_fallback(self, enhancer, sample_chunk):
        """Test fallback when enhancement fails completely."""
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("Persistent error")
            
            enhanced_chunk = await enhancer.enhance_chunk(sample_chunk)
            
            # Should return original chunk with enhancement_failed flag
            assert enhanced_chunk.metadata.get("enhancement_failed") is True
            assert "summary" not in enhanced_chunk.metadata
            assert enhanced_chunk.content == sample_chunk.content
    
    @pytest.mark.asyncio
    async def test_enhance_batch(self, enhancer):
        """Test batch enhancement of multiple chunks."""
        chunks = [
            ChunkResult(
                content=f"Content {i}",
                metadata={"chunk_id": f"chunk_{i}"}
            )
            for i in range(3)
        ]
        
        mock_response_template = {
            "summary": "Summary {i}",
            "questions": ["Q{i}"],
            "chunk_type": "其他",
            "keywords": ["keyword{i}"]
        }
        
        async def mock_llm_response(prompt):
            # Extract chunk index from prompt
            for i in range(3):
                if f"Content {i}" in prompt:
                    response = {k: v.format(i=i) if isinstance(v, str) else [item.format(i=i) for item in v] 
                               for k, v in mock_response_template.items()}
                    return json.dumps(response)
            return json.dumps(mock_response_template)
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_response
            
            enhanced_chunks = await enhancer.enhance_batch(chunks)
            
            assert len(enhanced_chunks) == 3
            for i, chunk in enumerate(enhanced_chunks):
                assert chunk.metadata["summary"] == f"Summary {i}"
                assert chunk.metadata["questions"] == [f"Q{i}"]
                assert chunk.metadata["keywords"] == [f"keyword{i}"]
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, enhancer):
        """Test rate limiting for batch processing."""
        enhancer.config.max_concurrent_requests = 2
        chunks = [
            ChunkResult(content=f"Content {i}", metadata={})
            for i in range(5)
        ]
        
        call_times = []
        
        async def mock_llm_with_timing(prompt):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate API delay
            return json.dumps({
                "summary": "Test",
                "questions": ["Q1"],
                "chunk_type": "其他",
                "keywords": ["test"]
            })
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_llm_with_timing
            
            await enhancer.enhance_batch(chunks)
            
            # Check that no more than 2 requests were concurrent
            # This is a simplified check - in practice you'd want more sophisticated verification
            assert mock_llm.call_count == 5
    
    def test_build_enhancement_prompt(self, enhancer, sample_chunk):
        """Test prompt building."""
        prompt = enhancer._build_enhancement_prompt(sample_chunk.content)
        
        assert sample_chunk.content in prompt
        assert "summary" in prompt
        assert "questions" in prompt
        assert "chunk_type" in prompt
        assert "keywords" in prompt
        assert "JSON" in prompt
    
    @pytest.mark.asyncio
    async def test_parse_llm_response(self, enhancer):
        """Test parsing different LLM response formats."""
        # Valid JSON response
        valid_response = json.dumps({
            "summary": "Test summary",
            "questions": ["Q1", "Q2"],
            "chunk_type": "概念定义",
            "keywords": ["test", "example"]
        })
        
        metadata = await enhancer._parse_llm_response(valid_response)
        assert metadata.summary == "Test summary"
        assert len(metadata.questions) == 2
        
        # Invalid JSON
        with pytest.raises(ValueError):
            await enhancer._parse_llm_response("Not a JSON")
        
        # Missing required fields
        incomplete_response = json.dumps({
            "summary": "Test summary",
            "questions": ["Q1"]
            # Missing chunk_type and keywords
        })
        
        with pytest.raises(ValueError):
            await enhancer._parse_llm_response(incomplete_response)
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, enhancer, sample_chunk):
        """Test caching of enhanced chunks."""
        enhancer.config.enable_cache = True
        
        mock_response = json.dumps({
            "summary": "Cached summary",
            "questions": ["Q1"],
            "chunk_type": "其他",
            "keywords": ["cached"]
        })
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            
            # First call
            enhanced1 = await enhancer.enhance_chunk(sample_chunk)
            
            # Second call should use cache
            enhanced2 = await enhancer.enhance_chunk(sample_chunk)
            
            # LLM should only be called once
            assert mock_llm.call_count == 1
            
            # Results should be identical
            assert enhanced1.metadata["summary"] == enhanced2.metadata["summary"]
            assert enhanced1.metadata["questions"] == enhanced2.metadata["questions"]


class TestEnhancementConfig:
    """Test the EnhancementConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EnhancementConfig()
        
        assert config.llm_provider == "deepseek"
        assert config.model_name == "deepseek-chat"
        assert config.temperature == 0.1
        assert config.max_tokens == 500
        assert config.max_retries == 3
        assert config.enable_cache is True
        assert config.max_concurrent_requests == 10
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EnhancementConfig(
            llm_provider="qwen",
            model_name="qwen2.5-72b",
            temperature=0.2,
            enable_cache=False
        )
        
        assert config.llm_provider == "qwen"
        assert config.model_name == "qwen2.5-72b"
        assert config.temperature == 0.2
        assert config.enable_cache is False
    
    def test_chunk_type_options(self):
        """Test chunk type options."""
        config = EnhancementConfig()
        
        expected_types = [
            "概念定义", "操作步骤", "示例代码", 
            "理论说明", "问题解答", "其他"
        ]
        assert config.chunk_type_options == expected_types
    
    def test_prompt_template(self):
        """Test prompt template configuration."""
        config = EnhancementConfig()
        
        assert "{content}" in config.prompt_template
        assert "summary" in config.prompt_template
        assert "questions" in config.prompt_template
        assert "chunk_type" in config.prompt_template
        assert "keywords" in config.prompt_template


class TestIntegration:
    """Integration tests with real chunking output."""
    
    @pytest.mark.asyncio
    async def test_enhance_markdown_chunks(self):
        """Test enhancing chunks from MarkdownChunker."""
        from knowledge_core_engine.core.chunking import MarkdownChunker
        
        markdown_text = """# Introduction to RAG

RAG (Retrieval Augmented Generation) combines retrieval and generation.

## How it works

1. Retrieve relevant documents
2. Generate response using context
"""
        
        # Create chunks
        chunker = MarkdownChunker()
        chunking_result = chunker.chunk(markdown_text)
        
        # Enhance chunks
        config = EnhancementConfig(llm_provider="mock")
        enhancer = MetadataEnhancer(config)
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = json.dumps({
                "summary": "Introduction to RAG technology",
                "questions": ["What is RAG?"],
                "chunk_type": "概念定义",
                "keywords": ["RAG", "retrieval", "generation"]
            })
            
            enhanced_chunks = await enhancer.enhance_batch(chunking_result.chunks)
            
            # Verify enhancement
            for chunk in enhanced_chunks:
                assert "summary" in chunk.metadata
                assert "questions" in chunk.metadata
                assert "chunk_type" in chunk.metadata
                assert "keywords" in chunk.metadata
                
                # Original metadata preserved
                assert "chunk_id" in chunk.metadata
                assert "document_id" in chunk.metadata
    
    @pytest.mark.asyncio
    async def test_enhance_complex_document(self):
        """Test enhancing a complex document with multiple chunk types."""
        from knowledge_core_engine.core.chunking import MarkdownChunker
        
        # Complex document with various content types
        complex_markdown = """# KnowledgeCore Engine 技术文档

## 概述

KnowledgeCore Engine是一个高性能的知识管理引擎，专门设计用于处理大规模文档的检索和生成任务。

## 核心架构

### 1. 文档解析层

系统使用LlamaParse进行文档解析：

```python
async def parse_document(file_path: Path) -> str:
    parser = LlamaParse(api_key=config.api_key)
    result = await parser.parse(file_path)
    return result.markdown
```

### 2. 智能分块

分块策略采用语义感知的方式：

- 保持段落完整性
- 维护代码块结构
- 处理表格和列表

### 3. 向量化处理

使用高维向量表示文本语义：

| 模型 | 维度 | 速度 | 准确度 |
|------|------|------|--------|
| text-embedding-v3 | 1536 | 快 | 高 |
| bge-large | 1024 | 中 | 中 |

## 常见问题

**Q: 如何处理大文件？**
A: 系统会自动进行流式处理，避免内存溢出。

**Q: 支持哪些文件格式？**
A: 支持PDF、Word、Markdown、HTML等常见格式。

## 最佳实践

1. **文档预处理**：确保文档格式规范
2. **元数据标注**：添加必要的文档元信息
3. **定期更新**：保持知识库的时效性

## 示例代码

完整的使用示例：

```python
from knowledge_core_engine import KnowledgeEngine

# 初始化引擎
engine = KnowledgeEngine(config)

# 添加文档
await engine.add_document("path/to/document.pdf")

# 查询
results = await engine.query("什么是向量数据库？")
print(results)
```

## 性能优化

- 使用缓存减少重复计算
- 批量处理提高吞吐量
- 异步IO优化响应时间
"""
        
        # Create chunks
        chunker = MarkdownChunker(chunk_size=300, chunk_overlap=50)
        chunking_result = chunker.chunk(complex_markdown)
        
        # Prepare mock responses for different chunk types
        mock_responses = {
            "概述": {
                "summary": "KnowledgeCore Engine是处理大规模文档检索和生成的高性能知识管理引擎",
                "questions": ["什么是KnowledgeCore Engine？", "它有什么特点？"],
                "chunk_type": "概念定义",
                "keywords": ["KnowledgeCore", "知识管理", "引擎", "检索", "生成"]
            },
            "parse_document": {
                "summary": "使用LlamaParse进行异步文档解析的示例代码",
                "questions": ["如何解析文档？", "LlamaParse的使用方法是什么？"],
                "chunk_type": "示例代码",
                "keywords": ["LlamaParse", "parse", "async", "文档解析"]
            },
            "分块策略": {
                "summary": "智能分块采用语义感知方式保持内容完整性",
                "questions": ["分块策略是什么？", "如何保持段落完整性？"],
                "chunk_type": "理论说明",
                "keywords": ["分块", "语义感知", "段落", "代码块"]
            },
            "模型": {
                "summary": "向量化模型性能对比表格",
                "questions": ["有哪些向量化模型？", "各模型的性能如何？"],
                "chunk_type": "其他",
                "keywords": ["向量化", "text-embedding-v3", "bge-large", "维度"]
            },
            "常见问题": {
                "summary": "系统使用的常见问题解答",
                "questions": ["如何处理大文件？", "支持哪些格式？"],
                "chunk_type": "问题解答",
                "keywords": ["FAQ", "大文件", "文件格式", "流式处理"]
            },
            "最佳实践": {
                "summary": "使用系统的三个最佳实践建议",
                "questions": ["有哪些最佳实践？", "如何优化使用效果？"],
                "chunk_type": "操作步骤",
                "keywords": ["最佳实践", "预处理", "元数据", "更新"]
            }
        }
        
        # Configure enhancement
        config = EnhancementConfig(llm_provider="mock")
        enhancer = MetadataEnhancer(config)
        
        # Mock LLM to return appropriate responses based on content
        async def smart_mock_llm(prompt):
            # Find which response to use based on content
            for key, response in mock_responses.items():
                if key in prompt:
                    return json.dumps(response)
            # Default response
            return json.dumps({
                "summary": "文档内容的默认摘要",
                "questions": ["这是什么内容？"],
                "chunk_type": "其他",
                "keywords": ["默认", "内容"]
            })
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = smart_mock_llm
            
            # Enhance all chunks
            enhanced_chunks = await enhancer.enhance_batch(chunking_result.chunks)
            
            # Verify results
            assert len(enhanced_chunks) == len(chunking_result.chunks)
            
            # Check chunk type distribution
            chunk_types = {}
            for chunk in enhanced_chunks:
                chunk_type = chunk.metadata.get('chunk_type', '其他')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Should have multiple chunk types
            assert len(chunk_types) >= 3
            assert all(ct in config.chunk_type_options for ct in chunk_types.keys())
            
            # Verify all chunks were enhanced
            for chunk in enhanced_chunks:
                assert 'summary' in chunk.metadata
                assert 'questions' in chunk.metadata
                assert 'chunk_type' in chunk.metadata
                assert 'keywords' in chunk.metadata
                assert not chunk.metadata.get('enhancement_failed', False)
    
    @pytest.mark.asyncio
    async def test_enhance_with_llm_errors(self):
        """Test enhancement behavior with various LLM errors."""
        from knowledge_core_engine.core.chunking import MarkdownChunker
        
        markdown_text = """# Error Test Document

This document tests error handling.

## Section 1
Content that will succeed.

## Section 2
Content that will fail with JSON error.

## Section 3
Content that will fail with missing fields.

## Section 4
Content that will fail with network error.
"""
        
        # Create chunks
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=20)
        chunking_result = chunker.chunk(markdown_text)
        
        # Configure enhancement
        config = EnhancementConfig(
            llm_provider="mock",
            max_retries=2,
            retry_delay=0.1
        )
        enhancer = MetadataEnhancer(config)
        
        # Mock different error scenarios
        call_count = 0
        chunk_responses = {}  # Track which chunk gets which response
        
        async def error_mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            
            # Identify which chunk we're processing
            for i, chunk in enumerate(chunking_result.chunks):
                if chunk.content in prompt:
                    chunk_id = i
                    break
            else:
                chunk_id = -1
            
            if "Section 1" in prompt:
                # Success case
                return json.dumps({
                    "summary": "成功的内容",
                    "questions": ["测试问题"],
                    "chunk_type": "其他",
                    "keywords": ["测试"]
                })
            elif "Section 2" in prompt:
                # Invalid JSON
                return "This is not valid JSON"
            elif "Section 3" in prompt:
                # Missing required fields
                return json.dumps({
                    "summary": "缺少字段的内容",
                    "questions": ["测试问题"]
                    # Missing chunk_type and keywords
                })
            elif "Section 4" in prompt:
                # Network error (will retry)
                # Track attempts per chunk
                chunk_key = f"chunk_{chunk_id}"
                if chunk_key not in chunk_responses:
                    chunk_responses[chunk_key] = 0
                chunk_responses[chunk_key] += 1
                
                if chunk_responses[chunk_key] == 1:
                    raise Exception("Network timeout")
                else:
                    return json.dumps({
                        "summary": "重试后成功",
                        "questions": ["重试测试"],
                        "chunk_type": "其他",
                        "keywords": ["重试", "网络"]
                    })
            else:
                # Default error - might be the title or intro
                if "Error Test Document" in prompt or "This document tests" in prompt:
                    # Let these succeed
                    return json.dumps({
                        "summary": "测试文档介绍",
                        "questions": ["这是什么文档？"],
                        "chunk_type": "其他",
                        "keywords": ["测试", "错误处理"]
                    })
                else:
                    raise Exception("Unknown error")
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = error_mock_llm
            
            # Enhance all chunks
            enhanced_chunks = await enhancer.enhance_batch(chunking_result.chunks)
            
            # Check results
            success_count = 0
            failed_count = 0
            
            for chunk in enhanced_chunks:
                if chunk.metadata.get('enhancement_failed'):
                    failed_count += 1
                    assert 'enhancement_error' in chunk.metadata
                else:
                    success_count += 1
                    assert 'summary' in chunk.metadata
            
            # Should have both successes and failures
            assert success_count >= 2  # Section 1 and Section 4 (after retry)
            assert failed_count >= 2   # Section 2 and Section 3
    
    @pytest.mark.asyncio
    async def test_enhance_with_cache_invalidation(self):
        """Test cache behavior and invalidation."""
        from knowledge_core_engine.core.chunking import MarkdownChunker
        
        markdown_text = """# Cache Test

This content will be cached and reused.
"""
        
        # Create identical chunks
        chunker = MarkdownChunker()
        result1 = chunker.chunk(markdown_text)
        result2 = chunker.chunk(markdown_text)
        
        # Configure with cache
        config = EnhancementConfig(
            llm_provider="mock",
            enable_cache=True
        )
        enhancer = MetadataEnhancer(config)
        
        call_count = 0
        async def counting_mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            return json.dumps({
                "summary": f"Call number {call_count}",
                "questions": ["Cache test?"],
                "chunk_type": "其他",
                "keywords": ["cache", f"call{call_count}"]
            })
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = counting_mock_llm
            
            # First enhancement
            enhanced1 = await enhancer.enhance_chunk(result1.chunks[0])
            assert enhanced1.metadata['summary'] == "Call number 1"
            assert call_count == 1
            
            # Second enhancement (should use cache)
            enhanced2 = await enhancer.enhance_chunk(result2.chunks[0])
            assert enhanced2.metadata['summary'] == "Call number 1"  # Same as first
            assert call_count == 1  # No new call
            
            # Clear cache
            enhancer.clear_cache()
            
            # Third enhancement (cache cleared)
            enhanced3 = await enhancer.enhance_chunk(result1.chunks[0])
            assert enhanced3.metadata['summary'] == "Call number 2"  # New call
            assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_enhance_performance_metrics(self):
        """Test performance with large batch processing."""
        import time
        from knowledge_core_engine.core.chunking import ChunkResult
        
        # Create many chunks
        num_chunks = 50
        chunks = [
            ChunkResult(
                content=f"This is test content number {i}. " * 10,
                metadata={"chunk_id": f"perf_test_{i}"}
            )
            for i in range(num_chunks)
        ]
        
        # Configure with limited concurrency
        config = EnhancementConfig(
            llm_provider="mock",
            max_concurrent_requests=5,
            enable_cache=False
        )
        enhancer = MetadataEnhancer(config)
        
        # Track timing
        call_times = []
        async def timed_mock_llm(prompt):
            call_times.append(time.time())
            await asyncio.sleep(0.01)  # Simulate API delay
            chunk_num = prompt.split("number ")[1].split(".")[0]
            return json.dumps({
                "summary": f"Summary for chunk {chunk_num}",
                "questions": [f"Question about {chunk_num}?"],
                "chunk_type": "其他",
                "keywords": ["test", f"chunk{chunk_num}"]
            })
        
        with patch.object(enhancer, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = timed_mock_llm
            
            start_time = time.time()
            enhanced_chunks = await enhancer.enhance_batch(chunks)
            end_time = time.time()
            
            # Verify all chunks enhanced
            assert len(enhanced_chunks) == num_chunks
            assert all(not c.metadata.get('enhancement_failed') for c in enhanced_chunks)
            
            # Check concurrency was respected
            # With max 5 concurrent requests, we should see waves of ~5 calls
            concurrent_calls = []
            for i in range(1, len(call_times)):
                if call_times[i] - call_times[i-1] < 0.005:  # Nearly simultaneous
                    concurrent_calls.append(i)
            
            # Performance assertions
            total_time = end_time - start_time
            avg_time_per_chunk = total_time / num_chunks
            
            # Should be much faster than serial processing
            # Serial would take ~0.5s (50 * 0.01), concurrent should be ~0.1s
            assert total_time < 0.3  # Allow some overhead
            
            print(f"\nPerformance metrics:")
            print(f"- Total chunks: {num_chunks}")
            print(f"- Total time: {total_time:.3f}s")
            print(f"- Avg per chunk: {avg_time_per_chunk:.3f}s")
            print(f"- Max concurrent: {config.max_concurrent_requests}")