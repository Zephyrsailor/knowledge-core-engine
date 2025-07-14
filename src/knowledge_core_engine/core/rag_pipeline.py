"""Unified RAG pipeline with configurable providers.

[EXPERIMENTAL] This module provides an alternative RAG pipeline implementation
that supports provider-based configuration and YAML config files.

Note: For most use cases, we recommend using KnowledgeEngine instead,
which provides a simpler and more feature-rich interface.

This module is kept for backward compatibility and advanced use cases
that require direct provider control.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import yaml
import os

from .providers import ProviderFactory, LLMProvider, EmbeddingProvider, VectorDBProvider
from .providers.vectordb import Document, QueryResult
from .chunking.base import ChunkResult


@dataclass
class PipelineConfig:
    """RAG pipeline configuration for provider-based pipeline.
    
    Note: This is different from core.config.RAGConfig.
    This config is specific to the provider-based pipeline implementation.
    """
    config_file: Optional[str] = None
    llm_provider: str = "deepseek"
    embedding_provider: str = "dashscope"
    vectordb_provider: str = "chromadb"
    embedding_strategy: str = "multi_vector"
    
    def __post_init__(self):
        if self.config_file and os.path.exists(self.config_file):
            self.load_from_file(self.config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.llm_provider = config.get('llm', {}).get('default', self.llm_provider)
        self.embedding_provider = config.get('embedding', {}).get('default', self.embedding_provider)
        self.vectordb_provider = config.get('vectordb', {}).get('default', self.vectordb_provider)
        self.embedding_strategy = config.get('app', {}).get('embedding_strategy', self.embedding_strategy)
        
        # Store full config for provider creation
        self._full_config = config
    
    def get_provider_config(self, provider_type: str, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        if hasattr(self, '_full_config'):
            return self._full_config.get(provider_type, {}).get('providers', {}).get(provider_name, {})
        return {"provider": provider_name}


class RAGPipeline:
    """Complete RAG pipeline with configurable providers.
    
    This is an alternative implementation to KnowledgeEngine that provides
    more direct control over providers but requires more manual configuration.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize RAG pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if not provided)
        """
        self.config = config or PipelineConfig()
        self._llm: Optional[LLMProvider] = None
        self._embedder: Optional[EmbeddingProvider] = None
        self._vectordb: Optional[VectorDBProvider] = None
    
    async def initialize(self):
        """Initialize all providers."""
        # Create LLM provider
        llm_config = self.config.get_provider_config("llm", self.config.llm_provider)
        self._llm = ProviderFactory.create("llm", llm_config)
        await self._llm.initialize()
        
        # Create embedding provider
        embedding_config = self.config.get_provider_config("embedding", self.config.embedding_provider)
        self._embedder = ProviderFactory.create("embedding", embedding_config)
        await self._embedder.initialize()
        
        # Create vector DB provider
        vectordb_config = self.config.get_provider_config("vectordb", self.config.vectordb_provider)
        self._vectordb = ProviderFactory.create("vectordb", vectordb_config)
        await self._vectordb.initialize()
        
        # Create collection
        dimension = self._embedder.get_dimension()
        await self._vectordb.create_collection(dimension)
    
    async def add_chunks(self, chunks: List[ChunkResult]):
        """Add chunks to the knowledge base.
        
        Args:
            chunks: List of chunks with content and metadata
        """
        documents = []
        
        for chunk in chunks:
            # Prepare text based on strategy
            if self.config.embedding_strategy == "multi_vector":
                text = self._embedder.prepare_text(chunk.content, chunk.metadata)
            else:
                text = chunk.content
            
            # Embed
            result = await self._embedder.embed(text)
            
            # Create document
            doc = Document(
                id=chunk.metadata.get("chunk_id", f"chunk_{hash(chunk.content)}"),
                embedding=result.embedding,
                text=chunk.content,  # Store original content
                metadata={
                    **chunk.metadata,
                    "embedding_model": result.model,
                    "embedding_strategy": self.config.embedding_strategy
                }
            )
            documents.append(doc)
        
        # Add to vector DB
        await self._vectordb.add_documents(documents)
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Query the knowledge base.
        
        Args:
            query: User query
            top_k: Number of results
            filter: Metadata filter
            
        Returns:
            List of relevant chunks
        """
        # Embed query
        query_result = await self._embedder.embed(query)
        
        # Search
        results = await self._vectordb.search(
            query_embedding=query_result.embedding,
            top_k=top_k,
            filter=filter
        )
        
        return results
    
    async def generate(
        self,
        query: str,
        contexts: List[QueryResult],
        include_citations: bool = True
    ) -> str:
        """Generate answer based on retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            include_citations: Whether to include source citations
            
        Returns:
            Generated answer
        """
        # Build prompt
        prompt = self._build_generation_prompt(query, contexts, include_citations)
        
        # Generate response
        response = await self._llm.complete(prompt)
        
        return response.content
    
    async def ask(
        self,
        question: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Complete RAG flow: retrieve and generate answer.
        
        Args:
            question: User question
            top_k: Number of contexts to retrieve
            filter: Metadata filter
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant contexts
        contexts = await self.query(question, top_k, filter)
        
        # Generate answer
        answer = await self.generate(question, contexts)
        
        return {
            "question": question,
            "answer": answer,
            "contexts": [
                {
                    "id": ctx.id,
                    "text": ctx.text[:200] + "...",
                    "score": ctx.score,
                    "metadata": ctx.metadata
                }
                for ctx in contexts
            ],
            "providers": {
                "llm": self.config.llm_provider,
                "embedding": self.config.embedding_provider,
                "vectordb": self.config.vectordb_provider
            }
        }
    
    def _build_generation_prompt(
        self,
        query: str,
        contexts: List[QueryResult],
        include_citations: bool
    ) -> str:
        """Build prompt for generation."""
        prompt_parts = [
            "基于以下相关文档内容回答用户问题。",
            "",
            "相关文档："
        ]
        
        for i, ctx in enumerate(contexts, 1):
            prompt_parts.append(f"\n[文档{i}] (相关度: {ctx.score:.2f})")
            prompt_parts.append(ctx.text)
            prompt_parts.append("")
        
        prompt_parts.extend([
            f"用户问题：{query}",
            "",
            "要求：",
            "1. 基于提供的文档内容准确回答",
            "2. 如果文档中没有相关信息，明确说明",
            "3. 保持客观和准确"
        ])
        
        if include_citations:
            prompt_parts.append("4. 在回答中标注信息来源，例如：[文档1]")
        
        prompt_parts.append("\n回答：")
        
        return "\n".join(prompt_parts)
    
    async def close(self):
        """Clean up resources."""
        if self._llm:
            await self._llm.close()
        if self._embedder:
            await self._embedder.close()
        if self._vectordb:
            await self._vectordb.close()


# --- Usage Example ---
"""
# Method 1: Use with configuration file
config = PipelineConfig(config_file="config/providers.yaml")
pipeline = RAGPipeline(config)
await pipeline.initialize()

# Method 2: Direct configuration
config = PipelineConfig(
    llm_provider="deepseek",
    embedding_provider="dashscope",
    vectordb_provider="chromadb",
    embedding_strategy="multi_vector"
)
pipeline = RAGPipeline(config)

# Method 3: Use KnowledgeEngine (Recommended)
from knowledge_core_engine import KnowledgeEngine
engine = KnowledgeEngine(
    llm_provider="deepseek",
    embedding_provider="dashscope"
)
await pipeline.initialize()

# Add knowledge
chunks = [
    ChunkResult(
        content="RAG combines retrieval and generation...",
        metadata={
            "chunk_id": "doc1_chunk1",
            "summary": "RAG技术介绍",
            "questions": ["什么是RAG?"],
            "source": "rag_guide.pdf"
        }
    )
]
await pipeline.add_chunks(chunks)

# Ask questions
result = await pipeline.ask("什么是RAG技术?")
print(result["answer"])

# Direct retrieval
contexts = await pipeline.query("RAG technology", top_k=5)

# Custom generation
answer = await pipeline.generate("Explain RAG", contexts)
"""