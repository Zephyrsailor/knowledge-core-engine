# Embedding Module Test Design

## Overview
The embedding module is responsible for converting enhanced text chunks into vector representations and storing them in ChromaDB for efficient retrieval.

## Test Structure

### 1. Unit Tests for Embedding Configuration
- Test default configuration values
- Test custom configuration
- Test API key loading from environment
- Test invalid configuration handling

### 2. Unit Tests for Text Embedder
- Test single text embedding
- Test batch text embedding
- Test handling of long texts (truncation)
- Test error handling (API failures)
- Test retry logic
- Test caching functionality

### 3. Unit Tests for Multi-Vector Strategy
- Test combining content + summary + questions
- Test weight distribution for different components
- Test handling missing metadata
- Test custom combination strategies

### 4. Unit Tests for ChromaDB Integration
- Test collection creation
- Test adding documents with metadata
- Test updating existing documents
- Test deletion of documents
- Test querying by vector similarity
- Test metadata filtering

### 5. Integration Tests
- Test end-to-end embedding pipeline
- Test embedding enhanced chunks from previous modules
- Test performance with large batches
- Test persistence and recovery

## Mock Strategy

1. **DashScope API Mock**
   - Mock embedding API responses
   - Simulate API errors and rate limits
   - Return consistent vectors for testing

2. **ChromaDB Mock**
   - Use in-memory ChromaDB for unit tests
   - Mock persistence operations
   - Simulate query results

## Test Data

1. **Sample Chunks**
   ```python
   sample_chunks = [
       ChunkResult(
           content="RAG technology combines retrieval and generation",
           metadata={
               "chunk_id": "test_1",
               "summary": "RAG combines retrieval with generation",
               "questions": ["What is RAG?", "How does RAG work?"],
               "chunk_type": "概念定义",
               "keywords": ["RAG", "retrieval", "generation"]
           }
       ),
       # More test chunks...
   ]
   ```

2. **Expected Behaviors**
   - Single chunk should produce one vector
   - Batch processing should be more efficient
   - Vectors should be 1536-dimensional (text-embedding-v3)
   - ChromaDB should store vectors with all metadata

## Performance Requirements
- Single embedding: < 100ms
- Batch of 100 embeddings: < 5s
- ChromaDB insertion: < 50ms per document
- Query response: < 200ms

## Error Scenarios
1. API key missing or invalid
2. DashScope API timeout
3. Rate limiting
4. ChromaDB connection failure
5. Disk space issues
6. Corrupted vector data

## Test Implementation Plan

1. **Phase 1**: Configuration and basic embedder tests
2. **Phase 2**: Multi-vector strategy tests
3. **Phase 3**: ChromaDB integration tests
4. **Phase 4**: End-to-end integration tests
5. **Phase 5**: Performance and stress tests