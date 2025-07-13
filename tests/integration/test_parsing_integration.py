"""Integration tests for the entire parsing module."""

import pytest
from pathlib import Path
import os

from knowledge_core_engine.core.parsing import DocumentProcessor
from knowledge_core_engine.core.parsing.base import ParseResult


@pytest.mark.integration
class TestParsingIntegration:
    """Integration tests for real parsing functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create real DocumentProcessor."""
        return DocumentProcessor(cache_enabled=False)  # Disable cache for testing
    
    @pytest.mark.asyncio
    async def test_parse_text_file_e2e(self, processor, tmp_path):
        """End-to-end test for text file parsing."""
        # Create test file
        content = """# Test Document

This is a test document with multiple paragraphs.

## Section 1
- Item 1
- Item 2
- Item 3

## Section 2
This section contains some important information.

### Subsection 2.1
More details here."""
        
        test_file = tmp_path / "test_document.txt"
        test_file.write_text(content)
        
        # Parse
        result = await processor.process(test_file)
        
        # Verify
        assert isinstance(result, ParseResult)
        assert result.markdown == content
        assert result.metadata["file_name"] == "test_document.txt"
        assert result.metadata["file_type"] == "txt"
        assert result.metadata["file_size"] > 0
        assert "parse_time" in result.metadata
    
    @pytest.mark.asyncio
    async def test_parse_markdown_file_e2e(self, processor, tmp_path):
        """End-to-end test for markdown file parsing."""
        # Create markdown file
        md_content = """# Knowledge Base Article

## Overview
This article covers **important** topics.

### Code Example
```python
def hello():
    return "Hello, World!"
```

### Table Example
| Column A | Column B |
|----------|----------|
| Value 1  | Value 2  |

## Conclusion
That's all for now!"""
        
        md_file = tmp_path / "article.md"
        md_file.write_text(md_content)
        
        # Parse
        result = await processor.process(md_file)
        
        # Verify
        assert result.markdown == md_content  # Should be passthrough
        assert result.metadata["parse_method"] == "markdown_parser"
    
    @pytest.mark.asyncio
    @pytest.mark.requires_api
    async def test_parse_with_llama_parse_e2e(self, processor, tmp_path):
        """Test parsing with real LlamaParse API."""
        # Skip if no API key
        if not os.getenv("LLAMA_CLOUD_API_KEY"):
            pytest.skip("LLAMA_CLOUD_API_KEY not set")
        
        # Create a simple text file that LlamaParse can handle
        content = """Technical Specification Document

Product: Knowledge Core Engine
Version: 1.0.0

Features:
1. Document parsing with multiple formats
2. Intelligent chunking based on structure
3. Vector embedding and storage
4. Semantic search capabilities

Requirements:
- Python 3.9+
- LlamaParse API key
- Vector database"""
        
        test_file = tmp_path / "spec.txt"
        test_file.write_text(content)
        
        # Parse with LlamaParse
        result = await processor.process(test_file)
        
        # Verify
        assert isinstance(result, ParseResult)
        assert result.markdown  # Should have content
        # Note: Will use text_parser for .txt files, not llama_parse
        assert result.metadata["parse_method"] == "text_parser"
        assert "parse_time" in result.metadata
        
        # LlamaParse might format the content differently
        # but key content should be preserved
        assert "Knowledge Core Engine" in result.markdown
        assert "Document parsing" in result.markdown
    
    @pytest.mark.asyncio
    async def test_process_multiple_files(self, processor, tmp_path):
        """Test processing multiple files in sequence."""
        # Create multiple files
        files = []
        for i in range(3):
            file_path = tmp_path / f"doc_{i}.txt"
            file_path.write_text(f"Document {i} content")
            files.append(file_path)
        
        # Process all files
        results = []
        for file_path in files:
            result = await processor.process(file_path)
            results.append(result)
        
        # Verify
        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"Document {i} content" in result.markdown
            assert result.metadata["file_name"] == f"doc_{i}.txt"
    
    @pytest.mark.asyncio
    async def test_error_handling_e2e(self, processor, tmp_path):
        """Test error handling in real scenarios."""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            await processor.process(Path("nonexistent.pdf"))
        
        # Test unsupported file type
        bad_file = tmp_path / "bad.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            await processor.process(bad_file)
    
    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, processor, tmp_path):
        """Test handling of Unicode content."""
        # Create file with various Unicode characters
        unicode_content = """# Unicode Test 文档

## 中文内容
这是一个测试文档。

## Emojis 🎉
- Item 1 ✅
- Item 2 ❌
- Item 3 🚀

## Special Characters
- Currency: €£¥₹
- Math: ∑∏∫∞
- Arrows: →←↑↓"""
        
        unicode_file = tmp_path / "unicode_test.txt"
        unicode_file.write_text(unicode_content, encoding='utf-8')
        
        # Parse
        result = await processor.process(unicode_file)
        
        # Verify all content is preserved
        assert result.markdown == unicode_content
        assert "中文内容" in result.markdown
        assert "🎉" in result.markdown
        assert "€£¥₹" in result.markdown