"""Example of using the RAG pipeline with different configurations."""

import asyncio
from pathlib import Path

from knowledge_core_engine.core.rag_pipeline import RAGPipeline, RAGConfig
from knowledge_core_engine.core.chunking import ChunkResult


async def example_basic_usage():
    """Basic RAG pipeline usage with default providers."""
    print("\n=== Basic RAG Usage ===")
    
    # Create pipeline with defaults
    pipeline = RAGPipeline()
    await pipeline.initialize()
    
    # Add some knowledge
    chunks = [
        ChunkResult(
            content="RAG (Retrieval Augmented Generation) 是一种结合了检索系统和生成模型的技术。"
                    "它通过先从知识库中检索相关信息，然后基于检索到的内容生成答案，"
                    "从而提供更准确、更有依据的回复。",
            metadata={
                "chunk_id": "rag_intro_1",
                "summary": "RAG技术将检索和生成结合，提供准确的AI回答",
                "questions": ["什么是RAG?", "RAG如何工作?", "RAG的优势是什么?"],
                "keywords": ["RAG", "检索增强生成", "AI技术"],
                "source": "rag_guide.md"
            }
        ),
        ChunkResult(
            content="RAG的主要优势包括：1) 减少幻觉：基于真实文档生成答案；"
                    "2) 可验证性：可以追溯答案来源；3) 动态更新：只需更新知识库，"
                    "无需重新训练模型；4) 成本效益：比微调大模型更经济。",
            metadata={
                "chunk_id": "rag_benefits_1",
                "summary": "RAG的四大优势：准确性、可验证、易更新、低成本",
                "questions": ["RAG有什么优势?", "为什么选择RAG?"],
                "keywords": ["RAG优势", "准确性", "成本效益"],
                "source": "rag_guide.md"
            }
        )
    ]
    
    await pipeline.add_chunks(chunks)
    print(f"Added {len(chunks)} chunks to knowledge base")
    
    # Ask questions
    questions = [
        "什么是RAG技术？",
        "RAG相比传统LLM有什么优势？",
        "如何实现一个RAG系统？"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = await pipeline.ask(question, top_k=3)
        print(f"A: {result['answer']}")
        print(f"   (Based on {len(result['contexts'])} sources)")
    
    await pipeline.close()


async def example_with_config_file():
    """Using RAG pipeline with configuration file."""
    print("\n=== RAG with Config File ===")
    
    # Load configuration from file
    config_path = Path("config/providers.yaml")
    if not config_path.exists():
        print(f"Config file {config_path} not found. Using defaults.")
        config = RAGConfig()
    else:
        config = RAGConfig(config_file=str(config_path))
        print(f"Loaded config from {config_path}")
        print(f"  LLM: {config.llm_provider}")
        print(f"  Embedding: {config.embedding_provider}")
        print(f"  VectorDB: {config.vectordb_provider}")
    
    pipeline = RAGPipeline(config)
    await pipeline.initialize()
    
    # Your knowledge operations...
    await pipeline.close()


async def example_custom_providers():
    """Using different provider combinations."""
    print("\n=== Custom Provider Combinations ===")
    
    # Example 1: DeepSeek + DashScope + ChromaDB (Default Chinese stack)
    config1 = RAGConfig(
        llm_provider="deepseek",
        embedding_provider="dashscope",
        vectordb_provider="chromadb"
    )
    
    # Example 2: OpenAI + OpenAI + Pinecone (Full OpenAI stack)
    config2 = RAGConfig(
        llm_provider="openai",
        embedding_provider="openai",
        vectordb_provider="pinecone"
    )
    
    # Example 3: Qwen + HuggingFace + ChromaDB (Mixed stack)
    config3 = RAGConfig(
        llm_provider="qwen",
        embedding_provider="huggingface",  # Local embedding
        vectordb_provider="chromadb"
    )
    
    print("Configuration examples created (not initialized)")


async def example_advanced_search():
    """Advanced search with filters and custom generation."""
    print("\n=== Advanced Search ===")
    
    pipeline = RAGPipeline()
    await pipeline.initialize()
    
    # Add diverse knowledge
    chunks = [
        ChunkResult(
            content="Python中的装饰器是一种设计模式...",
            metadata={
                "chunk_id": "py_decorator_1",
                "topic": "python",
                "difficulty": "intermediate",
                "content_type": "explanation"
            }
        ),
        ChunkResult(
            content="JavaScript的Promise用于处理异步操作...",
            metadata={
                "chunk_id": "js_promise_1",
                "topic": "javascript",
                "difficulty": "intermediate",
                "content_type": "explanation"
            }
        ),
        ChunkResult(
            content="深度学习中的Transformer架构...",
            metadata={
                "chunk_id": "ml_transformer_1",
                "topic": "machine_learning",
                "difficulty": "advanced",
                "content_type": "theory"
            }
        )
    ]
    
    await pipeline.add_chunks(chunks)
    
    # Search with filter
    print("\nSearching for Python content only:")
    contexts = await pipeline.query(
        "装饰器的使用",
        top_k=5,
        filter={"topic": "python"}
    )
    
    for ctx in contexts:
        print(f"- {ctx.id}: {ctx.text[:50]}... (score: {ctx.score:.3f})")
    
    # Custom generation
    if contexts:
        answer = await pipeline.generate(
            "解释Python装饰器",
            contexts,
            include_citations=True
        )
        print(f"\nGenerated answer: {answer}")
    
    await pipeline.close()


async def example_streaming_response():
    """Example with streaming LLM response."""
    print("\n=== Streaming Response ===")
    
    # Configure for streaming
    config = RAGConfig()
    config._full_config = {
        "llm": {
            "default": "deepseek",
            "providers": {
                "deepseek": {
                    "provider": "deepseek",
                    "stream": True  # Enable streaming
                }
            }
        }
    }
    
    pipeline = RAGPipeline(config)
    await pipeline.initialize()
    
    # Note: Actual streaming would require modifying the generate method
    # This is just to show the configuration approach
    
    await pipeline.close()


async def example_multi_language():
    """Example with multi-language content."""
    print("\n=== Multi-Language RAG ===")
    
    pipeline = RAGPipeline()
    await pipeline.initialize()
    
    # Add content in different languages
    chunks = [
        ChunkResult(
            content="Artificial Intelligence is transforming industries...",
            metadata={
                "language": "en",
                "chunk_id": "ai_en_1"
            }
        ),
        ChunkResult(
            content="人工智能正在改变各个行业...",
            metadata={
                "language": "zh",
                "chunk_id": "ai_zh_1"
            }
        ),
        ChunkResult(
            content="L'intelligence artificielle transforme les industries...",
            metadata={
                "language": "fr",
                "chunk_id": "ai_fr_1"
            }
        )
    ]
    
    await pipeline.add_chunks(chunks)
    
    # Query in different languages
    queries = [
        ("What is AI?", {"language": "en"}),
        ("什么是人工智能？", {"language": "zh"}),
        ("Qu'est-ce que l'IA?", {"language": "fr"})
    ]
    
    for query, filter in queries:
        print(f"\nQuery: {query}")
        contexts = await pipeline.query(query, top_k=2, filter=filter)
        print(f"Found {len(contexts)} results in {filter['language']}")
    
    await pipeline.close()


async def main():
    """Run all examples."""
    print("KnowledgeCore Engine - RAG Pipeline Examples")
    print("=" * 50)
    
    # Note: These are examples showing the API design
    # Actual execution would require:
    # 1. API keys configured
    # 2. Provider implementations completed
    # 3. Vector database initialized
    
    try:
        # Uncomment to run with actual implementation:
        # await example_basic_usage()
        # await example_with_config_file()
        # await example_custom_providers()
        # await example_advanced_search()
        # await example_streaming_response()
        # await example_multi_language()
        
        print("\nExamples demonstrate the flexible RAG pipeline design.")
        print("Configure your API keys to run with real providers.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())