"""
Advanced usage example for KnowledgeCore Engine.

This example demonstrates advanced features:
1. Batch document processing
2. Custom metadata enhancement
3. Advanced retrieval with filters
4. Multi-language support
5. Custom prompt templates
"""

import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.pipelines.ingestion import IngestionPipeline
from knowledge_core_engine.pipelines.retrieval import RetrievalPipeline
from knowledge_core_engine.pipelines.generation import GenerationPipeline
from knowledge_core_engine.core.generation.prompt_builder import PromptTemplate


async def batch_load_documents(directory: Path, config: RAGConfig):
    """Load multiple documents from a directory."""
    print(f"\n📁 Loading documents from: {directory}")
    
    ingestion = IngestionPipeline(config)
    await ingestion.initialize()
    
    # Find all supported documents
    supported_extensions = ['.pdf', '.md', '.txt', '.docx']
    documents = []
    for ext in supported_extensions:
        documents.extend(directory.glob(f"*{ext}"))
    
    print(f"📄 Found {len(documents)} documents to process")
    
    # Process documents in batch
    results = []
    for doc in documents:
        try:
            result = await ingestion.process_document(doc)
            results.append(result)
            print(f"✅ {doc.name}: {result['chunks_created']} chunks")
        except Exception as e:
            print(f"❌ {doc.name}: Failed - {e}")
    
    total_chunks = sum(r['chunks_created'] for r in results)
    print(f"\n📊 Total: {len(results)} documents, {total_chunks} chunks created")
    
    return results


async def advanced_retrieval_example(config: RAGConfig):
    """Demonstrate advanced retrieval features."""
    print("\n🔍 Advanced Retrieval Example")
    
    retrieval = RetrievalPipeline(config)
    await retrieval.initialize()
    
    # Example 1: Retrieval with metadata filters
    query = "RAG技术的优势"
    
    # Filter by document title
    filters = {
        "document_title": {"$eq": "RAG技术详解"}
    }
    
    contexts = await retrieval.retrieve(
        query=query,
        top_k=3,
        filters=filters
    )
    
    print(f"📚 Found {len(contexts)} contexts with filter")
    for ctx in contexts:
        print(f"   - Score: {ctx.score:.3f}, Doc: {ctx.metadata.get('document_title')}")
    
    # Example 2: Hybrid search with different weights
    contexts = await retrieval.retrieve(
        query=query,
        top_k=5,
        search_type="hybrid",
        hybrid_alpha=0.7  # 70% semantic, 30% keyword
    )
    
    print(f"\n📚 Hybrid search found {len(contexts)} contexts")
    
    # Example 3: Reranking
    if config.extra_params.get("enable_reranking", True):
        contexts = await retrieval.retrieve_with_rerank(
            query=query,
            top_k=10,
            rerank_top_k=3
        )
        print(f"🎯 After reranking: {len(contexts)} top contexts")
    
    return contexts


async def custom_generation_example(config: RAGConfig):
    """Demonstrate custom prompt templates and generation options."""
    print("\n✨ Custom Generation Example")
    
    # Create custom prompt template
    custom_template = PromptTemplate(
        name="technical_analysis",
        template="""作为技术专家，请基于以下技术文档回答问题。

问题：{query}

技术文档：
{contexts}

要求：
1. 提供技术性的深入分析
2. 使用专业术语并解释
3. 包含具体的技术细节
4. 如有必要，提供代码示例或配置示例
5. 在引用文档时使用[数字]标注

技术分析："""
    )
    
    # Setup pipelines
    retrieval = RetrievalPipeline(config)
    await retrieval.initialize()
    
    generation = GenerationPipeline(config)
    await generation.initialize()
    
    # Query with custom template
    query = "如何优化RAG系统的检索性能？"
    contexts = await retrieval.retrieve(query, top_k=5)
    
    result = await generation.generate(
        query=query,
        contexts=contexts,
        template=custom_template.template,
        enable_cot=True  # Enable chain of thought
    )
    
    print(f"💡 Technical Analysis:\n{result.answer}")
    
    # Example with few-shot learning
    few_shot_examples = [
        {
            "query": "什么是向量数据库？",
            "answer": "向量数据库是专门用于存储和检索高维向量数据的数据库系统[1]。它使用特殊的索引结构（如HNSW、IVF等）来实现高效的相似性搜索[2]。"
        }
    ]
    
    result = await generation.generate(
        query="解释什么是嵌入模型？",
        contexts=contexts,
        few_shot_examples=few_shot_examples
    )
    
    print(f"\n💡 Few-shot Answer:\n{result.answer}")


async def multi_language_example(config: RAGConfig):
    """Demonstrate multi-language support."""
    print("\n🌍 Multi-language Example")
    
    # English configuration
    en_config = RAGConfig(
        **config.dict(),
        extra_params={
            **config.extra_params,
            "language": "en"
        }
    )
    
    # Chinese configuration
    zh_config = RAGConfig(
        **config.dict(),
        extra_params={
            **config.extra_params,
            "language": "zh"
        }
    )
    
    # Process same query in different languages
    queries = {
        "en": "What are the advantages of RAG technology?",
        "zh": "RAG技术有哪些优势？"
    }
    
    for lang, query in queries.items():
        cfg = en_config if lang == "en" else zh_config
        
        retrieval = RetrievalPipeline(cfg)
        await retrieval.initialize()
        
        generation = GenerationPipeline(cfg)
        await generation.initialize()
        
        contexts = await retrieval.retrieve(query, top_k=3)
        result = await generation.generate(query, contexts)
        
        print(f"\n🗣️ Language: {lang.upper()}")
        print(f"❓ Query: {query}")
        print(f"💡 Answer: {result.answer[:200]}...")


async def performance_monitoring_example(config: RAGConfig):
    """Demonstrate performance monitoring and optimization."""
    print("\n📊 Performance Monitoring Example")
    
    import time
    
    # Initialize pipelines
    retrieval = RetrievalPipeline(config)
    generation = GenerationPipeline(config)
    
    # Measure initialization time
    start = time.time()
    await retrieval.initialize()
    await generation.initialize()
    init_time = time.time() - start
    print(f"⚡ Initialization time: {init_time:.2f}s")
    
    # Measure query performance
    query = "RAG系统的实施步骤是什么？"
    
    # Retrieval performance
    start = time.time()
    contexts = await retrieval.retrieve(query, top_k=5)
    retrieval_time = time.time() - start
    
    # Generation performance
    start = time.time()
    result = await generation.generate(query, contexts)
    generation_time = time.time() - start
    
    # Print metrics
    print(f"\n📈 Performance Metrics:")
    print(f"   - Retrieval: {retrieval_time:.3f}s ({len(contexts)} contexts)")
    print(f"   - Generation: {generation_time:.3f}s ({result.usage.get('total_tokens', 0)} tokens)")
    print(f"   - Total: {retrieval_time + generation_time:.3f}s")
    print(f"   - Tokens/sec: {result.usage.get('completion_tokens', 0) / generation_time:.1f}")
    
    # Memory usage
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"\n💾 Memory Usage: {memory_info.rss / 1024 / 1024:.1f} MB")


async def main():
    """Main function demonstrating advanced features."""
    print("=== KnowledgeCore Engine Advanced Usage ===")
    
    # Advanced configuration
    config = RAGConfig(
        # LLM Configuration
        llm_provider="deepseek",
        llm_model="deepseek-chat",
        llm_api_key=os.getenv("DEEPSEEK_API_KEY"),
        
        # Embedding Configuration
        embedding_provider="dashscope",
        embedding_model="text-embedding-v3",
        embedding_api_key=os.getenv("DASHSCOPE_API_KEY"),
        
        # Vector Store Configuration
        vector_store_provider="chromadb",
        vector_store_path="./data/chroma_db_advanced",
        
        # Advanced Settings
        temperature=0.1,
        max_tokens=2048,
        include_citations=True,
        
        # Extra Parameters
        extra_params={
            "language": "zh",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "enable_metadata_enhancement": True,
            "enable_reranking": True,
            "reranker_model": "bge-reranker-v2-m3",
            "citation_style": "inline",
            "enable_cot": True,
            "max_retries": 3,
            "temperature_decay": 0.1
        }
    )
    
    # Create sample documents directory
    docs_dir = Path("./data/sample_docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Run examples
    try:
        # 1. Batch document loading
        # await batch_load_documents(docs_dir, config)
        
        # 2. Advanced retrieval
        await advanced_retrieval_example(config)
        
        # 3. Custom generation
        await custom_generation_example(config)
        
        # 4. Multi-language support
        await multi_language_example(config)
        
        # 5. Performance monitoring
        await performance_monitoring_example(config)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Advanced examples completed!")


if __name__ == "__main__":
    asyncio.run(main())