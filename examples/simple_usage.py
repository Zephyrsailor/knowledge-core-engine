"""Simple usage example - demonstrating basic RAG functionality."""

import asyncio
import os
from pathlib import Path
from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.parsing.document_processor import DocumentProcessor
from knowledge_core_engine.core.chunking.pipeline import ChunkingPipeline
from knowledge_core_engine.core.embedding.embedder import TextEmbedder
from knowledge_core_engine.core.embedding.vector_store import VectorStore
from knowledge_core_engine.core.retrieval.retriever import Retriever
from knowledge_core_engine.core.generation.generator import Generator


async def main():
    # 1. Configure the system
    config = RAGConfig(
        llm_provider="deepseek",
        llm_api_key=os.getenv("DEEPSEEK_API_KEY", "your_api_key"),
        embedding_provider="dashscope",
        embedding_api_key=os.getenv("DASHSCOPE_API_KEY", "your_api_key"),
        vector_store_provider="chromadb",
        include_citations=True
    )
    
    # 2. Initialize components
    parser = DocumentProcessor()
    chunker = ChunkingPipeline(enable_smart_chunking=True)
    embedder = TextEmbedder(config)
    vector_store = VectorStore(config)
    retriever = Retriever(config)
    generator = Generator(config)
    
    await embedder.initialize()
    await vector_store.initialize()
    await retriever.initialize()
    await generator.initialize()
    
    # 3. Create a simple test document
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("""# RAG Technology Guide

## What is RAG?
RAG (Retrieval-Augmented Generation) combines retrieval and generation for better AI responses.
It first retrieves relevant information from a knowledge base, then generates accurate answers.

## Benefits of RAG
The main benefits of RAG include:
- Accuracy: Responses are grounded in retrieved documents
- Verifiability: Each answer can be traced to source documents
- Cost-effectiveness: No need to retrain models for new knowledge
- Flexibility: Knowledge base can be updated easily
""")
        doc_path = Path(f.name)
    
    # 4. Process the document
    print("📚 Processing document...")
    
    # Parse
    parse_result = await parser.process(doc_path)
    
    # Chunk
    chunk_result = await chunker.process_parse_result(parse_result)
    print(f"✅ Created {chunk_result.total_chunks} knowledge chunks")
    
    # Embed and store
    for chunk in chunk_result.chunks:
        embedding_result = await embedder.embed(chunk.content)
        await vector_store.add_documents([{
            "id": chunk.chunk_id,
            "content": chunk.content,
            "embedding": embedding_result.embedding,
            "metadata": chunk.metadata
        }])
    
    # 5. Ask questions
    query = "What is RAG and what are its benefits?"
    print(f"\n💡 Question: {query}")
    
    # Retrieve relevant contexts
    contexts = await retriever.retrieve(query, top_k=5)
    print(f"🔍 Found {len(contexts)} relevant contexts")
    
    # Generate answer
    answer = await generator.generate(query, contexts)
    print(f"\n📝 Answer: {answer.answer}")
    
    if answer.citations:
        print(f"\n📚 Citations:")
        for citation in answer.citations:
            print(f"   [{citation.index}] {citation.document_title}")
    
    # Clean up
    doc_path.unlink()


if __name__ == "__main__":
    # Set your API keys in environment
    # export DEEPSEEK_API_KEY=xxx
    # export DASHSCOPE_API_KEY=yyy
    
    asyncio.run(main())