# Embedding Module

A flexible and modular embedding system for converting text into vector representations.

## Overview

The embedding module is designed with modularity and flexibility in mind:

- **Multiple Providers**: Support for DashScope, OpenAI, HuggingFace, and custom providers
- **Flexible Strategies**: Various text preparation strategies for different use cases
- **Vector Store Abstraction**: Work with ChromaDB, Pinecone, Weaviate, or custom stores
- **Easy Integration**: Can be used standalone or as part of the larger system

## Key Concepts

### 1. Embedding Providers

Different services for converting text to vectors:

```python
from knowledge_core_engine.core.embedding import TextEmbedder, EmbeddingConfig

# DashScope (Alibaba Cloud)
config = EmbeddingConfig(
    provider="dashscope",
    model_name="text-embedding-v3"
)

# OpenAI
config = EmbeddingConfig(
    provider="openai", 
    model_name="text-embedding-3-large"
)

embedder = TextEmbedder(config)
result = await embedder.embed_text("Your text here")
```

### 2. Embedding Strategies

Different ways to prepare text before embedding:

#### Simple Strategy
Just embeds the raw content:
```python
strategy = create_strategy("simple")
# Input: "RAG combines retrieval and generation"
# Output: "RAG combines retrieval and generation"
```

#### Multi-Vector Strategy
Combines content with metadata for richer embeddings:
```python
strategy = create_strategy("multi_vector", {
    "include_summary": True,
    "include_questions": True
})

# Input: 
#   content: "RAG combines retrieval and generation"
#   metadata: {
#     "summary": "RAG技术结合检索和生成",
#     "questions": ["什么是RAG?", "RAG如何工作?"]
#   }
#
# Output:
#   "Content: RAG combines retrieval and generation
#    Summary: RAG技术结合检索和生成  
#    Questions: 什么是RAG? RAG如何工作?"
```

This enriched text creates embeddings that capture multiple semantic aspects, improving retrieval accuracy.

#### Hybrid Strategy
Weighted combination with emphasis control:
```python
strategy = create_strategy("hybrid", {
    "content_weight": 0.5,
    "summary_weight": 0.3,
    "questions_weight": 0.2
})
```

#### Custom Strategy
Define your own text preparation logic:
```python
def my_preparation(content, metadata):
    # Your custom logic
    return f"Custom: {content}"

strategy = create_strategy("custom", {
    "prepare_func": my_preparation
})
```

### 3. Vector Stores

Abstraction over different vector databases:

```python
from knowledge_core_engine.core.embedding import VectorStore, VectorStoreConfig

# ChromaDB
config = VectorStoreConfig(
    provider="chromadb",
    collection_name="my_knowledge"
)

# Pinecone
config = VectorStoreConfig(
    provider="pinecone",
    index_name="my_index"
)

store = VectorStore(config)
```

## Usage Examples

### Basic Usage

```python
from knowledge_core_engine.core.embedding import (
    TextEmbedder,
    EmbeddingConfig,
    create_strategy
)

# Configure
config = EmbeddingConfig(provider="dashscope")
embedder = TextEmbedder(config)

# Embed
result = await embedder.embed_text("Your text")
print(f"Vector dimension: {len(result.embedding)}")
```

### With Enhanced Metadata

```python
# Use with chunks from enhancement module
chunk = ChunkResult(
    content="RAG technology explanation...",
    metadata={
        "summary": "RAG combines retrieval and generation",
        "questions": ["What is RAG?", "How does it work?"],
        "keywords": ["RAG", "AI", "retrieval"]
    }
)

# Apply multi-vector strategy
strategy = create_strategy("multi_vector")
prepared_text = strategy.prepare_text(chunk.content, chunk.metadata)

# Embed the enriched text
result = await embedder.embed_text(prepared_text)
```

### Complete Pipeline

```python
from knowledge_core_engine.core.embedding import EmbeddingPipeline

# Configure pipeline
pipeline = EmbeddingPipeline(
    embedding_config={"provider": "dashscope"},
    vector_config={"provider": "chromadb"},
    strategy="multi_vector"
)

# Add documents
await pipeline.add_text(
    "Your content",
    metadata={"source": "doc1", "summary": "..."}
)

# Search
results = await pipeline.search("your query", top_k=5)
```

## Configuration

### Environment Variables

```bash
# API Keys
DASHSCOPE_API_KEY=your-key
OPENAI_API_KEY=your-key

# Model Settings
EMBEDDING_MODEL=text-embedding-v3
EMBEDDING_DIMENSIONS=1536

# Performance
EMBEDDING_BATCH_SIZE=25
EMBEDDING_CACHE_ENABLED=true
```

### Config File

```python
config = {
    "provider": "dashscope",
    "model_name": "text-embedding-v3",
    "dimensions": 1536,
    "batch_size": 25,
    "enable_cache": True,
    "strategy": {
        "name": "multi_vector",
        "include_summary": True,
        "include_questions": True
    }
}
```

## Advanced Features

### Batch Processing

```python
texts = ["text1", "text2", "text3", ...]
results = await embedder.embed_batch(texts)
```

### Caching

```python
config = EmbeddingConfig(enable_cache=True)
embedder = TextEmbedder(config)

# First call - hits API
result1 = await embedder.embed_text("test")

# Second call - from cache
result2 = await embedder.embed_text("test")
```

### Custom Providers

```python
from knowledge_core_engine.core.embedding import IEmbedder

class MyCustomEmbedder(IEmbedder):
    async def embed_text(self, text: str) -> EmbeddingResult:
        # Your implementation
        pass
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        # Your implementation
        pass

# Register and use
register_embedder("my_provider", MyCustomEmbedder)
```

## Best Practices

1. **Choose the Right Strategy**
   - Use `simple` for basic text search
   - Use `multi_vector` for knowledge bases with rich metadata
   - Use `hybrid` when you need fine control over component weights

2. **Batch When Possible**
   - Process multiple texts together for better performance
   - Configure appropriate batch sizes based on your provider limits

3. **Enable Caching**
   - Reduces API costs and improves performance
   - Especially useful during development and testing

4. **Monitor Usage**
   - Track token usage and costs
   - Set up alerts for rate limits

## Performance Tips

- **Batch Size**: Adjust based on provider limits (DashScope: 25, OpenAI: 100)
- **Concurrency**: Control with `max_concurrent_requests`
- **Caching**: Enable for frequently embedded texts
- **Text Length**: Most models have token limits, text is auto-truncated

## Error Handling

The module handles common errors gracefully:

- API failures with exponential backoff retry
- Rate limiting with automatic throttling
- Invalid inputs with clear error messages
- Network issues with configurable timeouts

## Extending the Module

### Add a New Provider

```python
@register_embedder("my_provider")
class MyEmbedder(BaseEmbedder):
    # Implementation
```

### Add a New Strategy

```python
@register_strategy("my_strategy")  
class MyStrategy(EmbeddingStrategy):
    # Implementation
```

### Add a New Vector Store

```python
@register_vector_store("my_store")
class MyVectorStore(BaseVectorStore):
    # Implementation
```

## Why Multi-Vector Strategy?

Traditional embedding only uses the raw content:
```
"RAG combines retrieval and generation" → [0.1, 0.2, ...]
```

Multi-vector strategy enriches the embedding:
```
"Content: RAG combines retrieval and generation
 Summary: RAG技术结合检索和生成
 Questions: 什么是RAG? RAG如何工作?" → [0.15, 0.25, ...]
```

Benefits:
- **Better Retrieval**: Questions help match user queries
- **Richer Context**: Summary captures essence
- **Multiple Perspectives**: Different ways to find the same content

This is especially useful for knowledge bases where users might search using different phrasings than the original content.