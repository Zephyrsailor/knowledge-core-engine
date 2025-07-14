"""Advanced retrieval example with multiple provider options.

This example demonstrates how to use the flexible provider system for
BM25 and reranking with different configurations.
"""

import asyncio
from knowledge_core_engine import KnowledgeEngine
from knowledge_core_engine.core.config import RAGConfig


async def example_1_local_highperformance():
    """Example 1: Local high-performance configuration.
    
    Suitable for machines with good hardware (e.g., 36GB RAM Mac).
    Uses local models for maximum performance.
    """
    print("=== Example 1: Local High-Performance Configuration ===")
    
    config = RAGConfig(
        # Use local Qwen3-8B reranker
        enable_reranking=True,
        reranker_provider="huggingface",
        reranker_model="qwen3-reranker-8b",
        use_fp16=True,  # Save memory with half precision
        
        # Use lightweight BM25S
        retrieval_strategy="hybrid",
        bm25_provider="bm25s",
        language="zh",  # Chinese support
        
        # LLM and embedding settings
        llm_provider="deepseek",
        embedding_provider="dashscope"
    )
    
    engine = KnowledgeEngine(**config.__dict__)
    
    # Add documents
    await engine.add("data/source_docs/")
    
    # Query with advanced retrieval
    answer = await engine.ask("什么是RAG技术的优势？")
    print(f"Answer: {answer}")
    print()


async def example_2_cloud_api():
    """Example 2: Cloud API configuration.
    
    Uses API services to save local resources.
    Good for lightweight deployments.
    """
    print("=== Example 2: Cloud API Configuration ===")
    
    config = RAGConfig(
        # Use DashScope API for reranking
        enable_reranking=True,
        reranker_provider="api",
        reranker_api_provider="dashscope",
        reranker_model="gte-rerank-v2",
        
        # Use lightweight BM25S
        retrieval_strategy="hybrid",
        bm25_provider="bm25s",
        
        # Other settings
        llm_provider="deepseek",
        embedding_provider="dashscope"
    )
    
    engine = KnowledgeEngine(**config.__dict__)
    
    # Add documents
    await engine.add("data/source_docs/")
    
    # Query
    answer = await engine.ask("Explain the benefits of RAG")
    print(f"Answer: {answer}")
    print()


async def example_3_enterprise_elasticsearch():
    """Example 3: Enterprise configuration with Elasticsearch.
    
    Uses Elasticsearch for BM25 and BGE for reranking.
    Suitable for production deployments.
    """
    print("=== Example 3: Enterprise Elasticsearch Configuration ===")
    
    config = RAGConfig(
        # Use BGE reranker (smaller model)
        enable_reranking=True,
        reranker_provider="huggingface",
        reranker_model="bge-reranker-v2-m3",
        use_fp16=True,
        
        # Use Elasticsearch for BM25
        retrieval_strategy="hybrid",
        bm25_provider="elasticsearch",
        elasticsearch_url="http://localhost:9200",
        elasticsearch_index="knowledge_core",
        
        # Other settings
        llm_provider="deepseek",
        embedding_provider="dashscope"
    )
    
    try:
        engine = KnowledgeEngine(**config.__dict__)
        
        # Add documents
        await engine.add("data/source_docs/")
        
        # Query
        answer = await engine.ask("What are the key components of RAG?")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Note: This example requires Elasticsearch to be running.")
        print(f"Error: {e}")
    print()


async def example_4_minimal_vector_only():
    """Example 4: Minimal vector-only configuration.
    
    No BM25, no reranking. Simple and fast.
    """
    print("=== Example 4: Minimal Vector-Only Configuration ===")
    
    config = RAGConfig(
        # Disable advanced features
        enable_reranking=False,
        retrieval_strategy="vector",  # Vector search only
        bm25_provider="none",
        
        # Basic settings
        llm_provider="deepseek",
        embedding_provider="dashscope"
    )
    
    engine = KnowledgeEngine(**config.__dict__)
    
    # Add documents
    await engine.add("data/source_docs/")
    
    # Query
    answer = await engine.ask("What is RAG?")
    print(f"Answer: {answer}")
    print()


async def example_5_custom_configuration():
    """Example 5: Custom configuration with specific models."""
    print("=== Example 5: Custom Configuration ===")
    
    # You can mix and match different providers
    config = RAGConfig(
        # Use Cohere API for reranking
        enable_reranking=True,
        reranker_provider="api",
        reranker_api_provider="cohere",
        reranker_model="rerank-english-v2.0",
        
        # Use BM25S with custom parameters
        retrieval_strategy="hybrid",
        bm25_provider="bm25s",
        bm25_k1=1.2,
        bm25_b=0.75,
        
        # Adjust weights for hybrid search
        vector_weight=0.6,
        bm25_weight=0.4,
        
        # Enable query expansion
        enable_query_expansion=True,
        query_expansion_method="llm",
        query_expansion_count=3,
        
        # Other settings
        llm_provider="deepseek",
        embedding_provider="dashscope",
        retrieval_top_k=20,
        rerank_top_k=5
    )
    
    try:
        engine = KnowledgeEngine(**config.__dict__)
        
        # Add documents
        await engine.add("data/source_docs/")
        
        # Query
        answer = await engine.ask("How does hybrid retrieval improve RAG performance?")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Note: This example requires Cohere API key.")
        print(f"Error: {e}")
    print()


async def main():
    """Run all examples."""
    # Example 1: Local high-performance (recommended for your 36GB Mac)
    await example_1_local_highperformance()
    
    # Example 2: Cloud API (save resources)
    await example_2_cloud_api()
    
    # Example 3: Enterprise with Elasticsearch
    await example_3_enterprise_elasticsearch()
    
    # Example 4: Minimal configuration
    await example_4_minimal_vector_only()
    
    # Example 5: Custom configuration
    await example_5_custom_configuration()


if __name__ == "__main__":
    # Note: Make sure you have configured your API keys in .env file
    asyncio.run(main())