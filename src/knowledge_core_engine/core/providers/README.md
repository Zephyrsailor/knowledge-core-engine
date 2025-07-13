# Provider System Design

## Overview

The provider system allows flexible configuration of LLM, Embedding, and VectorDB components through a unified interface.

## Key Features

### 1. **Unified Interface**
All providers follow the same pattern:
```python
# Create any provider type with configuration
provider = ProviderFactory.create("provider_type", config)
await provider.initialize()
```

### 2. **Configuration Flexibility**

#### From Dictionary
```python
config = {
    "provider": "deepseek",
    "api_key": "sk-xxx",
    "model": "deepseek-chat"
}
llm = ProviderFactory.create("llm", config)
```

#### From YAML File
```yaml
llm:
  default: deepseek
  providers:
    deepseek:
      provider: deepseek
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat
```

#### From Environment
```bash
export DEEPSEEK_API_KEY=sk-xxx
export DASHSCOPE_API_KEY=sk-yyy
```

### 3. **Easy Provider Switching**

Change providers without changing code:
```python
# Just change configuration
config.llm_provider = "openai"  # was "deepseek"
config.embedding_provider = "openai"  # was "dashscope"

# Same code works
pipeline = RAGPipeline(config)
```

## Supported Providers

### LLM Providers
- **DeepSeek**: Cost-effective Chinese LLM
- **Qwen**: Alibaba's Tongyi Qianwen
- **OpenAI**: GPT models (and compatible APIs)

### Embedding Providers
- **DashScope**: Alibaba Cloud embeddings
- **OpenAI**: OpenAI embeddings
- **HuggingFace**: Local embeddings

### Vector Database Providers
- **ChromaDB**: Local/persistent vector storage
- **Pinecone**: Cloud vector database
- More can be added easily

## Adding New Providers

### 1. Create Provider Class
```python
class MyLLMProvider(LLMProvider):
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        # Your implementation
        pass
```

### 2. Register Provider
```python
ProviderFactory.register("llm", "myllm", MyLLMProvider)
```

### 3. Use It
```python
config = {"provider": "myllm", "api_key": "..."}
llm = ProviderFactory.create("llm", config)
```

## Configuration Examples

### Minimal Configuration
```python
# Uses defaults and environment variables
pipeline = RAGPipeline()
```

### Custom Configuration
```python
config = RAGConfig(
    llm_provider="qwen",
    embedding_provider="dashscope",
    vectordb_provider="chromadb"
)
pipeline = RAGPipeline(config)
```

### Production Configuration
```yaml
# config/production.yaml
llm:
  default: deepseek
  providers:
    deepseek:
      provider: deepseek
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-chat
      temperature: 0.1
      timeout: 60
      max_retries: 5

embedding:
  default: dashscope
  providers:
    dashscope:
      provider: dashscope
      api_key: ${DASHSCOPE_API_KEY}
      model: text-embedding-v3
      batch_size: 25
      
vectordb:
  default: chromadb
  providers:
    chromadb:
      provider: chromadb
      persist_directory: /data/vectors
      collection_name: production_kb
```

## Multi-Vector Strategy

The embedding provider includes built-in support for multi-vector strategy:

```python
# Automatic multi-vector preparation
text = "RAG combines retrieval and generation"
metadata = {
    "summary": "RAG技术结合检索和生成",
    "questions": ["什么是RAG?", "RAG如何工作?"]
}

# Provider prepares text automatically
prepared = embedder.prepare_text(text, metadata)
# Result: "Content: RAG combines...\nSummary: RAG技术...\nQuestions: ..."
```

## Best Practices

1. **Use Environment Variables for Secrets**
   ```bash
   export DEEPSEEK_API_KEY=sk-xxx
   export DASHSCOPE_API_KEY=sk-yyy
   ```

2. **Configure Once, Use Everywhere**
   ```python
   # Load config once
   config = RAGConfig(config_file="config/providers.yaml")
   
   # Use in multiple places
   pipeline1 = RAGPipeline(config)
   pipeline2 = RAGPipeline(config)
   ```

3. **Provider-Specific Features**
   ```python
   # Access provider-specific features through extra_params
   config = {
       "provider": "openai",
       "extra_params": {
           "organization": "org-xxx",
           "request_timeout": 120
       }
   }
   ```

4. **Graceful Fallbacks**
   ```python
   try:
       # Try primary provider
       config.llm_provider = "deepseek"
       pipeline = RAGPipeline(config)
   except Exception:
       # Fallback to secondary
       config.llm_provider = "qwen"
       pipeline = RAGPipeline(config)
   ```

## Performance Considerations

1. **Batching**: Embedding providers automatically batch requests
2. **Caching**: Enable caching to avoid redundant API calls
3. **Local Options**: Use HuggingFace for offline embeddings
4. **Persistence**: ChromaDB supports persistent storage

## Testing Different Providers

```python
# Test different combinations
configs = [
    {"llm": "deepseek", "embedding": "dashscope"},
    {"llm": "openai", "embedding": "openai"},
    {"llm": "qwen", "embedding": "huggingface"}
]

for cfg in configs:
    config = RAGConfig(**cfg)
    pipeline = RAGPipeline(config)
    # Test your use case
```