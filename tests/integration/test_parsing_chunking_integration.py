"""Integration test for parsing and chunking pipeline."""

import pytest
from pathlib import Path
import tempfile
import asyncio

from knowledge_core_engine.core.parsing.document_processor import DocumentProcessor
from knowledge_core_engine.core.chunking import ChunkingPipeline, MarkdownChunker, SmartChunker


class TestParsingChunkingIntegration:
    """Test the integration between parsing and chunking modules."""
    
    @pytest.mark.asyncio
    async def test_markdown_document_pipeline(self):
        """Test processing a Markdown document through parsing and chunking."""
        # Create a test Markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""# Technical Documentation

## Introduction
This is a technical document about Python programming.

## Installation
To install Python, follow these steps:
1. Download Python from python.org
2. Run the installer
3. Verify installation with `python --version`

## Code Example
Here's a simple Python function:

```python
def hello_world():
    print("Hello, World!")
    return True
```

## Conclusion
Python is a versatile programming language.
""")
            temp_path = Path(f.name)
        
        try:
            # Parse the document
            processor = DocumentProcessor()
            parse_result = await processor.process(temp_path)
            
            assert parse_result.markdown is not None
            assert "# Technical Documentation" in parse_result.markdown
            
            # Chunk the parsed content
            chunking_pipeline = ChunkingPipeline(enable_smart_chunking=True)
            chunking_result = await chunking_pipeline.process_parse_result(parse_result)
            
            assert chunking_result.total_chunks > 0
            assert len(chunking_result.chunks) == chunking_result.total_chunks
            
            # Verify chunks preserve important content
            all_content = " ".join(chunk.content for chunk in chunking_result.chunks)
            assert "Python programming" in all_content
            assert "hello_world" in all_content
            
            # Verify metadata propagation
            for chunk in chunking_result.chunks:
                assert 'source' in chunk.metadata
                assert chunk.metadata['source'] == str(temp_path)
            
            # Create LlamaIndex nodes
            nodes = chunking_pipeline.create_llama_index_nodes(chunking_result)
            assert len(nodes) == chunking_result.total_chunks
            
            # Verify node metadata
            for i, node in enumerate(nodes):
                assert node.id_ == f"{chunking_result.document_metadata.get('document_id', 'doc_3a14aef0')}_chunk_{i}"
                assert node.metadata['chunk_index'] == i
                assert node.metadata['total_chunks'] == chunking_result.total_chunks
            
        finally:
            # Clean up
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_content_type_detection(self):
        """Test that different content types are properly detected and chunked."""
        test_cases = [
            # Q&A format
            ("""Q: What is Python?
A: Python is a high-level programming language.

Q: Why use Python?
A: Python is easy to learn and has a large ecosystem.""", "qa"),
            
            # Technical documentation with code
            ("""# API Documentation

## Function: process_data

```python
def process_data(input_list):
    return [x * 2 for x in input_list]
```

This function doubles each element in the input list.""", "technical"),
            
            # Dialogue format
            ("""Speaker 1: How do we implement this feature?
Speaker 2: We should use the strategy pattern.
Speaker 1: That sounds good. Can you show an example?""", "dialogue")
        ]
        
        processor = DocumentProcessor()
        smart_chunker = SmartChunker()
        
        for content, expected_type in test_cases:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(content)
                temp_path = Path(f.name)
            
            try:
                # Parse
                parse_result = await processor.process(temp_path)
                
                # Chunk with smart chunker
                chunking_result = smart_chunker.chunk(
                    parse_result.markdown,
                    parse_result.metadata
                )
                
                # Verify content type detection
                assert chunking_result.document_metadata['content_type'] == expected_type
                
            finally:
                temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test processing multiple documents in batch."""
        # Create multiple test files
        temp_files = []
        
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(f"""# Document {i}

## Section 1
Content for document {i}, section 1.

## Section 2
Content for document {i}, section 2.
""")
                temp_files.append(Path(f.name))
        
        try:
            # Parse all documents
            processor = DocumentProcessor()
            parse_results = []
            for temp_file in temp_files:
                result = await processor.process(temp_file)
                parse_results.append(result)
            
            assert len(parse_results) == 3
            for i, result in enumerate(parse_results):
                assert f"Document {i}" in result.markdown
            
            # Chunk all documents
            chunking_pipeline = ChunkingPipeline()
            chunking_results = await chunking_pipeline.process_batch(parse_results)
            
            assert len(chunking_results) == 3
            for i, result in enumerate(chunking_results):
                assert result.total_chunks > 0
                # Verify content preservation
                all_content = " ".join(chunk.content for chunk in result.chunks)
                assert f"Document {i}" in all_content
            
        finally:
            # Clean up
            for temp_file in temp_files:
                temp_file.unlink()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in the integration."""
        # Test with non-existent file
        processor = DocumentProcessor()
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            await processor.process(Path("/non/existent/file.md"))
        
        # Test with unsupported file type
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                await processor.process(temp_path)
        finally:
            temp_path.unlink()
    
    def test_chunking_statistics(self):
        """Test that we can get useful statistics from chunking."""
        # Create a document with known structure
        content = """# Main Title

## Section 1
This is a paragraph with exactly 100 characters to test chunk size calculations accurately and reliably.

## Section 2
Short section.

## Section 3
This is another paragraph that is designed to be exactly 150 characters long to ensure we can test our chunk size statistics properly and accurately verify them.
"""
        
        markdown_chunker = MarkdownChunker(chunk_size=200, chunk_overlap=0)
        result = markdown_chunker.chunk(content)
        
        # Get statistics
        pipeline = ChunkingPipeline()
        stats = pipeline.get_chunking_stats(result)
        
        assert stats['total_chunks'] > 0
        assert stats['average_chunk_size'] > 0
        assert stats['min_chunk_size'] > 0
        assert stats['max_chunk_size'] >= stats['min_chunk_size']
        assert stats['total_characters'] == sum(len(c.content) for c in result.chunks)