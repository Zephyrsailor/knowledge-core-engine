"""Demo of metadata enhancement using LLM."""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_core_engine.core.chunking import MarkdownChunker
from knowledge_core_engine.core.enhancement import MetadataEnhancer, EnhancementConfig


async def main():
    """Run metadata enhancement demo."""
    
    # Sample document
    markdown_text = """# RAG技术详解

## 什么是RAG？

RAG（Retrieval Augmented Generation）是一种结合了检索系统和生成模型的AI技术。它通过先检索相关信息，
然后基于检索到的内容生成回答，从而提供更准确、更有依据的响应。

## RAG的工作原理

### 1. 文档处理阶段
首先，系统需要处理和索引大量的文档：
- 文档解析：将各种格式的文档转换为统一格式
- 文档分块：将长文档切分为适合检索的小块
- 向量化：使用嵌入模型将文本转换为向量

### 2. 检索阶段
当用户提出问题时：
```python
# 1. 将问题转换为向量
query_vector = embedding_model.encode(user_query)

# 2. 在向量数据库中搜索相似内容
similar_chunks = vector_db.search(query_vector, top_k=5)

# 3. 获取相关文档内容
contexts = [chunk.text for chunk in similar_chunks]
```

### 3. 生成阶段
基于检索到的内容生成答案：
- 构建包含上下文的提示词
- 调用大语言模型生成回答
- 确保回答基于检索到的信息

## RAG的优势

1. **准确性提升**：基于实际文档内容，减少幻觉
2. **可验证性**：可以追溯答案的来源
3. **灵活更新**：只需更新文档库，无需重新训练模型
4. **成本效益**：相比微调模型，成本更低

## 应用场景

RAG技术广泛应用于：
- 企业知识库问答
- 客户服务系统
- 技术文档查询
- 法律文档分析
"""
    
    print("=== RAG Document Chunking and Enhancement Demo ===\n")
    
    # Step 1: Chunk the document
    print("Step 1: Chunking document...")
    chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
    chunking_result = chunker.chunk(markdown_text, metadata={
        "source": "rag_guide.md",
        "document_id": "rag_doc_001"
    })
    
    print(f"Created {chunking_result.total_chunks} chunks\n")
    
    # Step 2: Enhance chunks with metadata
    print("Step 2: Enhancing chunks with LLM metadata...")
    
    # Configure enhancement (using mock for demo)
    config = EnhancementConfig(
        llm_provider="mock",  # Use mock for demo
        temperature=0.1,
        enable_cache=True
    )
    
    enhancer = MetadataEnhancer(config)
    
    # Enhance all chunks
    enhanced_chunks = await enhancer.enhance_batch(chunking_result.chunks)
    
    print(f"Enhanced {len(enhanced_chunks)} chunks\n")
    
    # Step 3: Display results
    print("Step 3: Results\n")
    print("-" * 80)
    
    for i, chunk in enumerate(enhanced_chunks[:3]):  # Show first 3 chunks
        print(f"\n=== Chunk {i + 1} ===")
        print(f"Chunk ID: {chunk.metadata.get('chunk_id', 'N/A')}")
        print(f"Content Type: {chunk.metadata.get('content_type', 'N/A')}")
        print(f"Hierarchy Path: {chunk.metadata.get('hierarchy_path', 'N/A')}")
        
        print(f"\nContent Preview:")
        print(f"{chunk.content[:150]}..." if len(chunk.content) > 150 else chunk.content)
        
        print(f"\nEnhanced Metadata:")
        print(f"- Summary: {chunk.metadata.get('summary', 'N/A')}")
        print(f"- Chunk Type: {chunk.metadata.get('chunk_type', 'N/A')}")
        print(f"- Keywords: {', '.join(chunk.metadata.get('keywords', []))}")
        
        print(f"\n- Potential Questions:")
        for j, question in enumerate(chunk.metadata.get('questions', []), 1):
            print(f"  {j}. {question}")
        
        print("-" * 80)
    
    # Step 4: Show statistics
    print("\n=== Enhancement Statistics ===")
    
    # Count chunk types
    chunk_types = {}
    for chunk in enhanced_chunks:
        chunk_type = chunk.metadata.get('chunk_type', '其他')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    print("\nChunk Type Distribution:")
    for chunk_type, count in chunk_types.items():
        print(f"- {chunk_type}: {count}")
    
    # Calculate average metrics
    avg_keywords = sum(len(c.metadata.get('keywords', [])) for c in enhanced_chunks) / len(enhanced_chunks)
    avg_questions = sum(len(c.metadata.get('questions', [])) for c in enhanced_chunks) / len(enhanced_chunks)
    
    print(f"\nAverage keywords per chunk: {avg_keywords:.1f}")
    print(f"Average questions per chunk: {avg_questions:.1f}")
    
    # Check for enhancement failures
    failed_chunks = [c for c in enhanced_chunks if c.metadata.get('enhancement_failed')]
    if failed_chunks:
        print(f"\nFailed enhancements: {len(failed_chunks)}")
    else:
        print("\nAll chunks enhanced successfully!")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())