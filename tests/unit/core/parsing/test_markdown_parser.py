"""Tests for markdown parser."""

import pytest
from pathlib import Path

from knowledge_core_engine.core.parsing.parsers.markdown_parser import MarkdownParser
from knowledge_core_engine.core.parsing.base import ParseResult


class TestMarkdownParser:
    """Test MarkdownParser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create MarkdownParser instance."""
        return MarkdownParser()
    
    @pytest.mark.asyncio
    async def test_parse_markdown_file(self, parser, tmp_path):
        """Test parsing markdown file."""
        # Given
        markdown_content = """# Main Title

## Section 1
This is the first section with **bold** and *italic* text.

### Subsection 1.1
- Item 1
- Item 2
- Item 3

## Section 2
A table example:

| Column A | Column B |
|----------|----------|
| Value 1  | Value 2  |

```python
def hello():
    print("Hello, World!")
```
"""
        
        md_file = tmp_path / "test.md"
        md_file.write_text(markdown_content, encoding='utf-8')
        
        # When
        result = await parser.parse(md_file)
        
        # Then
        assert isinstance(result, ParseResult)
        assert result.markdown == markdown_content
        assert result.metadata["file_name"] == "test.md"
        assert result.metadata["file_type"] == "md"
        assert result.metadata["parse_method"] == "markdown_parser"
        assert result.metadata["encoding"] == "utf-8"
    
    @pytest.mark.asyncio
    async def test_parse_markdown_passthrough(self, parser, tmp_path):
        """Test that markdown content is passed through unchanged."""
        # Given
        original_content = """# Complex Markdown

[Link](https://example.com)

![Image](image.png)

> Blockquote

1. Ordered
2. List

- [ ] Task 1
- [x] Task 2
"""
        
        md_file = tmp_path / "complex.md"
        md_file.write_text(original_content)
        
        # When
        result = await parser.parse(md_file)
        
        # Then
        assert result.markdown == original_content
    
    @pytest.mark.asyncio
    async def test_parse_empty_markdown(self, parser, tmp_path):
        """Test parsing empty markdown file."""
        # Given
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        
        # When
        result = await parser.parse(empty_file)
        
        # Then
        assert result.markdown == ""
        assert result.metadata["file_size"] == 0
    
    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self, parser):
        """Test parsing non-existent file."""
        # Given
        non_existent = Path("does_not_exist.md")
        
        # When/Then
        with pytest.raises(FileNotFoundError):
            await parser.parse(non_existent)