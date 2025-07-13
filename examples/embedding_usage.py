"""Examples of using the embedding module in different scenarios.

This demonstrates the flexibility and modularity of the embedding system.
"""

import asyncio
from typing import List, Dict, Any

# Import from our module
from knowledge_core_engine.core.embedding import (
    EmbeddingConfig,
    TextEmbedder,
    EmbeddingStrategy,
    create_strategy,
    VectorStore,
    VectorStoreConfig
)
from knowledge_core_engine.core.chunking import ChunkResult


async def example_1_simple_embedding():
    """Example 1: Simple text embedding with default settings."""
    print("\n=== Example 1: Simple Embedding ===")
    
    # Configure embedder
    config = EmbeddingConfig(
        provider="dashscope",
        model_name="text-embedding-v3",
        api_key="your-api-key"  # In practice, load from env
    )
    
    # Create embedder
    embedder = TextEmbedder(config)
    
    # Embed a single text
    text = "RAG technology combines retrieval and generation for better AI responses."
    result = await embedder.embed_text(text)
    
    print(f"Text: {text}")
    print(f"Embedding dimension: {len(result.embedding)}")
    print(f"Model used: {result.model}")


async def example_2_batch_embedding():
    """Example 2: Batch embedding with caching."""
    print("\n=== Example 2: Batch Embedding ===")
    
    # Configure with caching enabled
    config = EmbeddingConfig(
        provider="dashscope",
        enable_cache=True,
        batch_size=10
    )
    
    embedder = TextEmbedder(config)
    
    # Batch embed multiple texts
    texts = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are neural networks?",
        "Explain transformer architecture."
    ]
    
    results = await embedder.embed_batch(texts)
    
    print(f"Embedded {len(results)} texts")
    for i, result in enumerate(results):
        print(f"  Text {i+1}: {result.text[:50]}...")


async def example_3_multi_vector_strategy():
    """Example 3: Using multi-vector strategy with enhanced metadata."""
    print("\n=== Example 3: Multi-Vector Strategy ===")
    
    # Create a chunk with rich metadata (from enhancement module)
    chunk = ChunkResult(
        content="RAG (Retrieval Augmented Generation) is a technique that combines "
                "retrieval systems with language models to provide accurate, "
                "context-aware responses.",
        metadata={
            "chunk_id": "doc1_chunk1",
            "summary": "RAG combines retrieval and generation for better AI responses",
            "questions": [
                "What is RAG?",
                "How does RAG work?",
                "What are the benefits of RAG?"
            ],
            "keywords": ["RAG", "retrieval", "generation", "AI"],
            "chunk_type": "概念定义"
        }
    )
    
    # Configure multi-vector strategy
    strategy = create_strategy("multi_vector", {
        "include_summary": True,
        "include_questions": True,
        "include_keywords": True
    })
    
    # Prepare text using strategy
    prepared_text = strategy.prepare_text(chunk.content, chunk.metadata)
    
    print("Original content:")
    print(f"  {chunk.content}")
    print("\nPrepared text for embedding:")
    print(f"  {prepared_text}")
    
    # Now embed the prepared text
    embedder = TextEmbedder(EmbeddingConfig())
    result = await embedder.embed_text(prepared_text)
    print(f"\nEmbedding created with dimension: {len(result.embedding)}")


async def example_4_custom_strategy():
    """Example 4: Creating a custom embedding strategy."""
    print("\n=== Example 4: Custom Strategy ===")
    
    # Define a custom preparation function
    def prepare_for_qa(content: str, metadata: Dict[str, Any]) -> str:
        """Custom function that formats text for Q&A systems."""
        parts = []
        
        # Add context
        if metadata.get("document_title"):
            parts.append(f"Document: {metadata['document_title']}")
        
        # Format as Q&A if questions are available
        if metadata.get("questions"):
            for q in metadata["questions"]:
                parts.append(f"Q: {q}")
            parts.append(f"A: {content}")
        else:
            parts.append(content)
        
        return "\n".join(parts)
    
    # Create custom strategy
    strategy = create_strategy("custom", {
        "name": "qa_format",
        "prepare_func": prepare_for_qa
    })
    
    # Use it
    chunk_metadata = {
        "document_title": "RAG技术指南",
        "questions": ["什么是RAG技术?", "RAG有什么优势?"]
    }
    
    prepared = strategy.prepare_text(
        "RAG是一种先进的AI技术...",
        chunk_metadata
    )
    
    print("Custom formatted text:")
    print(prepared)


async def example_5_vector_store_integration():
    """Example 5: Complete pipeline with vector storage."""
    print("\n=== Example 5: Vector Store Integration ===")
    
    # Configure vector store
    vector_config = VectorStoreConfig(
        provider="chromadb",
        collection_name="my_knowledge_base",
        persist_directory="./my_vectors"
    )
    
    vector_store = VectorStore(vector_config)
    
    # Configure embedder with strategy
    embedder = TextEmbedder(EmbeddingConfig())
    strategy = create_strategy("multi_vector")
    
    # Process chunks
    chunks = [
        ChunkResult(
            content="Content 1...",
            metadata={"chunk_id": "1", "summary": "Summary 1"}
        ),
        ChunkResult(
            content="Content 2...",
            metadata={"chunk_id": "2", "summary": "Summary 2"}
        )
    ]
    
    # Embed and store
    for chunk in chunks:
        # Prepare text with strategy
        prepared_text = strategy.prepare_text(chunk.content, chunk.metadata)
        
        # Embed
        embedding_result = await embedder.embed_text(prepared_text)
        
        # Store in vector database
        doc = {
            "id": chunk.metadata["chunk_id"],
            "embedding": embedding_result.embedding,
            "text": chunk.content,
            "metadata": chunk.metadata
        }
        vector_store.add([doc])
    
    print(f"Stored {len(chunks)} chunks in vector database")
    
    # Query example
    query = "What is RAG?"
    query_embedding = await embedder.embed_text(query)
    
    results = vector_store.query(
        query_embedding=query_embedding.embedding,
        top_k=5
    )
    
    print(f"\nQuery results for '{query}':")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result['score']:.3f}, ID: {result['id']}")


async def example_6_different_providers():
    """Example 6: Using different embedding providers."""
    print("\n=== Example 6: Different Providers ===")
    
    # DashScope (Alibaba Cloud)
    dashscope_config = EmbeddingConfig(
        provider="dashscope",
        model_name="text-embedding-v3",
        dimensions=1536
    )
    
    # OpenAI
    openai_config = EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-large",
        dimensions=3072
    )
    
    # HuggingFace local model
    hf_config = EmbeddingConfig(
        provider="huggingface",
        model_name="BAAI/bge-large-zh-v1.5",
        dimensions=1024
    )
    
    # The same interface works for all providers
    for config in [dashscope_config, openai_config, hf_config]:
        print(f"\nProvider: {config.provider}")
        print(f"Model: {config.model_name}")
        print(f"Dimensions: {config.dimensions}")
        
        # Same code works with any provider
        embedder = TextEmbedder(config)
        # result = await embedder.embed_text("Test text")


async def example_7_modular_usage():
    """Example 7: Using embedding module in your own project."""
    print("\n=== Example 7: Modular Usage ===")
    
    # Your project can use just the parts you need
    
    # Option 1: Just embedding
    from knowledge_core_engine.core.embedding import TextEmbedder, EmbeddingConfig
    
    embedder = TextEmbedder(EmbeddingConfig(provider="dashscope"))
    
    # Option 2: Just strategies
    from knowledge_core_engine.core.embedding import create_strategy
    
    strategy = create_strategy("multi_vector")
    prepared = strategy.prepare_text("content", {"summary": "sum"})
    
    # Option 3: Complete pipeline
    from knowledge_core_engine.core.embedding import EmbeddingPipeline
    
    pipeline = EmbeddingPipeline(
        embedding_config={"provider": "dashscope"},
        vector_config={"provider": "chromadb"},
        strategy="multi_vector"
    )
    
    # Simple API
    await pipeline.add_text("Your text", metadata={"source": "doc1"})
    results = await pipeline.search("query text", top_k=5)
    
    print("Embedding module can be used flexibly in your project!")


def main():
    """Run all examples."""
    print("KnowledgeCore Engine - Embedding Module Examples")
    print("=" * 50)
    
    # Note: These are examples, actual execution would need real API keys
    print("\nThese examples demonstrate the API design.")
    print("To run with real embeddings, configure your API keys.")
    
    # Uncomment to run with real implementation:
    # asyncio.run(example_1_simple_embedding())
    # asyncio.run(example_2_batch_embedding())
    # asyncio.run(example_3_multi_vector_strategy())
    # asyncio.run(example_4_custom_strategy())
    # asyncio.run(example_5_vector_store_integration())
    # asyncio.run(example_6_different_providers())
    # asyncio.run(example_7_modular_usage())


if __name__ == "__main__":
    main()