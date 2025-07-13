"""
Basic usage example for KnowledgeCore Engine.

This example demonstrates the complete RAG pipeline:
1. Load and process documents
2. Build a knowledge base
3. Ask questions and get answers with citations
"""

import asyncio
import os
from pathlib import Path
from typing import List

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.pipelines.ingestion import IngestionPipeline
from knowledge_core_engine.pipelines.retrieval import RetrievalPipeline
from knowledge_core_engine.pipelines.generation import GenerationPipeline


async def load_documents(file_path: Path, config: RAGConfig):
    """Load and process a document into the knowledge base."""
    print(f"\n📄 Loading document: {file_path}")
    
    # Initialize ingestion pipeline
    ingestion = IngestionPipeline(config)
    await ingestion.initialize()
    
    # Process the document
    try:
        result = await ingestion.process_document(file_path)
        print(f"✅ Successfully processed: {result['chunks_created']} chunks created")
        print(f"   Document ID: {result['document_id']}")
        return result
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        raise


async def query_knowledge_base(query: str, config: RAGConfig):
    """Query the knowledge base and get an answer."""
    print(f"\n🔍 Query: {query}")
    
    # Initialize retrieval pipeline
    retrieval = RetrievalPipeline(config)
    await retrieval.initialize()
    
    # Retrieve relevant contexts
    contexts = await retrieval.retrieve(query, top_k=5)
    print(f"📚 Found {len(contexts)} relevant contexts")
    
    # Initialize generation pipeline
    generation = GenerationPipeline(config)
    await generation.initialize()
    
    # Generate answer
    result = await generation.generate(query, contexts)
    
    print(f"\n💡 Answer:\n{result.answer}")
    
    if result.citations:
        print(f"\n📖 Citations:")
        for citation in result.citations:
            print(f"   [{citation.index}] {citation.document_title} (p.{citation.page})")
    
    print(f"\n📊 Tokens used: {result.usage.get('total_tokens', 'N/A')}")
    
    return result


async def main():
    """Main example demonstrating complete RAG workflow."""
    print("=== KnowledgeCore Engine Usage Example ===")
    
    # Configuration
    config = RAGConfig(
        # LLM Configuration
        llm_provider="deepseek",  # or "qwen", "openai"
        llm_model="deepseek-chat",
        llm_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Set your API key
        
        # Embedding Configuration
        embedding_provider="dashscope",  # Using Qwen embeddings
        embedding_model="text-embedding-v3",
        embedding_api_key=os.getenv("DASHSCOPE_API_KEY"),  # Set your API key
        
        # Vector Store Configuration
        vector_store_provider="chromadb",
        vector_store_path="./data/chroma_db",
        
        # Generation Settings
        temperature=0.1,  # Low temperature for factual answers
        max_tokens=2048,
        include_citations=True,
        
        # Extra Parameters
        extra_params={
            "language": "zh",  # Chinese language
            "chunk_size": 512,
            "chunk_overlap": 50,
            "enable_metadata_enhancement": True,
            "citation_style": "inline"
        }
    )
    
    # Example 1: Load a document
    print("\n📚 Example 1: Loading Documents")
    
    # Create sample document if it doesn't exist
    sample_doc = Path("./data/sample_rag_intro.md")
    if not sample_doc.exists():
        sample_doc.parent.mkdir(parents=True, exist_ok=True)
        sample_doc.write_text("""
# RAG技术详解

## 什么是RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合了信息检索和文本生成的AI技术。它通过在生成答案之前先从知识库中检索相关信息，从而大大提高了语言模型的准确性和可靠性。

## RAG的核心优势

### 1. 减少幻觉
传统的大语言模型容易产生"幻觉"，即生成看似合理但实际错误的信息。RAG通过基于真实文档生成答案，显著降低了这种风险。

### 2. 知识可更新
与需要重新训练的传统模型不同，RAG系统的知识库可以随时更新，无需修改底层模型。

### 3. 可解释性强
RAG生成的每个答案都可以追溯到具体的源文档，提供了清晰的引用链，增强了答案的可信度。

### 4. 成本效益高
相比于训练专门的领域模型，RAG使用现有的通用模型配合领域知识库，大大降低了部署成本。

## RAG的工作流程

1. **文档处理**：将原始文档解析、分块，并转换为向量表示
2. **检索阶段**：根据用户查询，从向量数据库中检索最相关的文档块
3. **生成阶段**：将检索到的上下文与查询一起输入LLM，生成准确的答案
4. **引用标注**：在答案中标注信息来源，提供可追溯性

## 企业应用场景

- **知识管理系统**：构建企业内部知识问答系统
- **客户服务**：基于产品文档的智能客服
- **法律咨询**：基于法规文档的合规性查询
- **医疗辅助**：基于医学文献的诊疗建议

## 实施RAG的关键考虑

实施RAG系统需要考虑多个技术要素：
- 文档解析的准确性
- 分块策略的优化
- 向量化模型的选择
- 检索算法的调优
- 生成模型的配置
- 整体系统的性能优化

## 总结

RAG技术代表了AI应用的一个重要方向，它通过结合检索和生成的优势，为企业提供了一种实用、可靠、可解释的AI解决方案。随着技术的不断发展，RAG将在更多领域发挥重要作用。
""", encoding="utf-8")
    
    # Load the document
    await load_documents(sample_doc, config)
    
    # Example 2: Ask questions
    print("\n💬 Example 2: Asking Questions")
    
    questions = [
        "什么是RAG技术？它的全称是什么？",
        "RAG相比传统大语言模型有哪些优势？",
        "企业可以在哪些场景使用RAG技术？",
        "实施RAG系统需要考虑哪些技术要素？"
    ]
    
    for question in questions:
        await query_knowledge_base(question, config)
        print("\n" + "="*50)
    
    # Example 3: Streaming generation (if needed)
    print("\n🌊 Example 3: Streaming Answer Generation")
    
    query = "详细说明RAG的工作流程"
    retrieval = RetrievalPipeline(config)
    await retrieval.initialize()
    contexts = await retrieval.retrieve(query, top_k=3)
    
    generation = GenerationPipeline(config)
    await generation.initialize()
    
    print(f"🔍 Query: {query}")
    print("💡 Answer (streaming):")
    
    accumulated_answer = ""
    async for chunk in generation.stream_generate(query, contexts):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            accumulated_answer += chunk.content
        
        if chunk.is_final and chunk.citations:
            print(f"\n\n📖 Citations:")
            for citation in chunk.citations:
                print(f"   [{citation.index}] {citation.document_title}")
    
    print("\n\n✨ Example completed!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())