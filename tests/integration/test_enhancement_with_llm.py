"""Integration tests for metadata enhancement with real LLM providers.

These tests are skipped by default as they require actual API keys.
To run these tests, set the environment variable: INTEGRATION_TESTS=true
"""

import pytest
import os
import asyncio
from typing import List

from knowledge_core_engine.core.chunking import MarkdownChunker
from knowledge_core_engine.core.enhancement import MetadataEnhancer, EnhancementConfig


# Skip these tests unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "true",
    reason="Integration tests require INTEGRATION_TESTS=true"
)


class TestDeepSeekIntegration:
    """Test metadata enhancement with DeepSeek API."""
    
    @pytest.fixture
    def deepseek_config(self):
        """Create config for DeepSeek."""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")
        
        return EnhancementConfig(
            llm_provider="deepseek",
            model_name="deepseek-chat",
            api_key=api_key,
            temperature=0.1,
            max_tokens=500
        )
    
    @pytest.mark.asyncio
    async def test_enhance_real_document(self, deepseek_config):
        """Test enhancing a real document with DeepSeek."""
        # Sample document
        markdown_text = """# Understanding Large Language Models

## Introduction

Large Language Models (LLMs) have revolutionized natural language processing 
by demonstrating remarkable capabilities in understanding and generating human-like text.

## Key Components

### 1. Architecture
Most modern LLMs are based on the Transformer architecture, which uses 
self-attention mechanisms to process sequences of text.

### 2. Training Process
LLMs are trained on massive datasets using unsupervised learning techniques:
- Pre-training on large text corpora
- Fine-tuning for specific tasks
- Reinforcement Learning from Human Feedback (RLHF)

## Applications

LLMs are used in various applications:
- Text generation and completion
- Question answering systems
- Code generation
- Language translation
- Sentiment analysis

## Challenges

Despite their capabilities, LLMs face several challenges:
- Hallucination and factual accuracy
- Computational resources requirements
- Bias in training data
- Context length limitations
"""
        
        # Create chunks
        chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
        chunking_result = chunker.chunk(markdown_text)
        
        # Enhance with real LLM
        enhancer = MetadataEnhancer(deepseek_config)
        enhanced_chunks = await enhancer.enhance_batch(chunking_result.chunks[:3])  # Test first 3
        
        # Verify results
        for chunk in enhanced_chunks:
            assert not chunk.metadata.get('enhancement_failed', False)
            assert 'summary' in chunk.metadata
            assert 'questions' in chunk.metadata
            assert 'chunk_type' in chunk.metadata
            assert 'keywords' in chunk.metadata
            
            # Check quality
            assert len(chunk.metadata['summary']) > 10
            assert len(chunk.metadata['questions']) >= 1
            assert chunk.metadata['chunk_type'] in deepseek_config.chunk_type_options
            assert len(chunk.metadata['keywords']) >= 2
            
            print(f"\nChunk enhanced successfully:")
            print(f"Summary: {chunk.metadata['summary']}")
            print(f"Type: {chunk.metadata['chunk_type']}")
            print(f"Keywords: {', '.join(chunk.metadata['keywords'])}")


class TestQwenIntegration:
    """Test metadata enhancement with Qwen API."""
    
    @pytest.fixture
    def qwen_config(self):
        """Create config for Qwen."""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not set")
        
        return EnhancementConfig(
            llm_provider="qwen",
            model_name="qwen2.5-72b-instruct",
            api_key=api_key,
            temperature=0.1,
            max_tokens=500
        )
    
    @pytest.mark.asyncio
    async def test_enhance_chinese_document(self, qwen_config):
        """Test enhancing a Chinese document with Qwen."""
        # Chinese document
        markdown_text = """# 知识图谱技术概述

## 什么是知识图谱？

知识图谱是一种结构化的知识表示方法，通过实体、关系和属性来描述现实世界中的概念和它们之间的联系。

## 核心技术

### 1. 知识抽取
从非结构化或半结构化数据中提取实体、关系和属性：
- 命名实体识别（NER）
- 关系抽取
- 属性抽取

### 2. 知识融合
将来自不同数据源的知识进行整合：
- 实体对齐
- 实体消歧
- 知识合并

### 3. 知识推理
基于已有知识推导新知识：
- 基于规则的推理
- 基于统计的推理
- 基于深度学习的推理

## 应用场景

知识图谱在多个领域有广泛应用：
- 智能搜索和问答系统
- 推荐系统
- 金融风控
- 医疗诊断辅助
"""
        
        # Create chunks
        chunker = MarkdownChunker(chunk_size=300, chunk_overlap=50)
        chunking_result = chunker.chunk(markdown_text)
        
        # Enhance with Qwen
        enhancer = MetadataEnhancer(qwen_config)
        enhanced_chunks = await enhancer.enhance_batch(chunking_result.chunks)
        
        # Verify results
        for chunk in enhanced_chunks:
            assert not chunk.metadata.get('enhancement_failed', False)
            
            # Check Chinese content handling
            assert any('\u4e00' <= char <= '\u9fff' for char in chunk.metadata['summary'])
            assert all(len(q) > 5 for q in chunk.metadata['questions'])
            
            print(f"\n中文内容增强成功:")
            print(f"摘要: {chunk.metadata['summary']}")
            print(f"类型: {chunk.metadata['chunk_type']}")


class TestPerformanceComparison:
    """Compare performance between different LLM providers."""
    
    @pytest.mark.asyncio
    async def test_provider_comparison(self):
        """Compare DeepSeek and Qwen performance."""
        import time
        
        # Skip if no API keys
        if not (os.getenv("DEEPSEEK_API_KEY") and os.getenv("DASHSCOPE_API_KEY")):
            pytest.skip("Both API keys required for comparison")
        
        # Test document
        test_doc = """# Performance Testing Document

This document contains various types of content to test LLM performance.

## Code Example

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## Technical Explanation

The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones.

## Table Data

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Recursive | O(2^n)         | O(n)             |
| Dynamic   | O(n)           | O(n)             |
| Iterative | O(n)           | O(1)             |
"""
        
        # Create chunks
        chunker = MarkdownChunker(chunk_size=200, chunk_overlap=30)
        chunking_result = chunker.chunk(test_doc)
        
        # Test DeepSeek
        deepseek_config = EnhancementConfig(
            llm_provider="deepseek",
            model_name="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
        deepseek_enhancer = MetadataEnhancer(deepseek_config)
        
        start = time.time()
        deepseek_results = await deepseek_enhancer.enhance_batch(chunking_result.chunks)
        deepseek_time = time.time() - start
        
        # Test Qwen
        qwen_config = EnhancementConfig(
            llm_provider="qwen",
            model_name="qwen2.5-72b-instruct",
            api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        qwen_enhancer = MetadataEnhancer(qwen_config)
        
        start = time.time()
        qwen_results = await qwen_enhancer.enhance_batch(chunking_result.chunks)
        qwen_time = time.time() - start
        
        # Compare results
        print("\n=== Performance Comparison ===")
        print(f"DeepSeek time: {deepseek_time:.2f}s")
        print(f"Qwen time: {qwen_time:.2f}s")
        print(f"Chunks processed: {len(chunking_result.chunks)}")
        
        # Compare quality (subjective)
        for i, (ds, qw) in enumerate(zip(deepseek_results, qwen_results)):
            print(f"\n--- Chunk {i+1} ---")
            print(f"DeepSeek summary: {ds.metadata.get('summary', 'N/A')}")
            print(f"Qwen summary: {qw.metadata.get('summary', 'N/A')}")


class TestErrorHandling:
    """Test error handling with real API calls."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test handling of rate limits."""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            pytest.skip("DEEPSEEK_API_KEY not set")
        
        # Create many chunks to trigger rate limits
        chunks = []
        for i in range(50):
            chunk = MarkdownChunker().chunk(f"Test content {i}").chunks[0]
            chunks.append(chunk)
        
        # Configure with aggressive concurrency
        config = EnhancementConfig(
            llm_provider="deepseek",
            api_key=api_key,
            max_concurrent_requests=20,  # High concurrency
            max_retries=3,
            retry_delay=1.0
        )
        
        enhancer = MetadataEnhancer(config)
        results = await enhancer.enhance_batch(chunks)
        
        # Should handle rate limits gracefully
        success_count = sum(1 for r in results if not r.metadata.get('enhancement_failed'))
        print(f"\nProcessed {success_count}/{len(chunks)} chunks successfully")
        assert success_count > 0  # At least some should succeed


if __name__ == "__main__":
    # Run specific test manually
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "deepseek":
        asyncio.run(TestDeepSeekIntegration().test_enhance_real_document(None))
    elif len(sys.argv) > 1 and sys.argv[1] == "qwen":
        asyncio.run(TestQwenIntegration().test_enhance_chinese_document(None))
    else:
        print("Usage: python test_enhancement_with_llm.py [deepseek|qwen]")