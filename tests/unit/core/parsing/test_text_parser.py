"""Tests for text parser."""

import pytest
from pathlib import Path

from knowledge_core_engine.core.parsing.parsers.text_parser import TextParser
from knowledge_core_engine.core.parsing.base import ParseResult


class TestTextParser:
    """Test TextParser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create TextParser instance."""
        return TextParser()
    
    @pytest.mark.asyncio
    async def test_parse_text_file(self, parser, tmp_path):
        """Test parsing plain text file."""
        # Given
        text_content = """This is a test document.
It has multiple lines.

And even paragraphs!"""
        
        text_file = tmp_path / "test.txt"
        text_file.write_text(text_content, encoding='utf-8')
        
        # When
        result = await parser.parse(text_file)
        
        # Then
        assert isinstance(result, ParseResult)
        assert result.markdown == text_content
        assert result.metadata["file_name"] == "test.txt"
        assert result.metadata["file_type"] == "txt"
        assert result.metadata["file_size"] == len(text_content.encode('utf-8'))
        assert result.metadata["parse_method"] == "text_parser"
        assert result.metadata["encoding"] == "utf-8"
        assert "parse_time" in result.metadata
    
    @pytest.mark.asyncio
    async def test_parse_empty_file(self, parser, tmp_path):
        """Test parsing empty text file."""
        # Given
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        # When
        result = await parser.parse(empty_file)
        
        # Then
        assert result.markdown == ""
        assert result.metadata["file_size"] == 0
    
    @pytest.mark.asyncio
    async def test_parse_utf8_with_special_chars(self, parser, tmp_path):
        """Test parsing text with special UTF-8 characters."""
        # Given
        special_content = "Special chars: €£¥ 中文 日本語 🎉"
        text_file = tmp_path / "special.txt"
        text_file.write_text(special_content, encoding='utf-8')
        
        # When
        result = await parser.parse(text_file)
        
        # Then
        assert result.markdown == special_content
    
    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self, parser):
        """Test parsing non-existent file."""
        # Given
        non_existent = Path("does_not_exist.txt")
        
        # When/Then
        with pytest.raises(FileNotFoundError):
            await parser.parse(non_existent)
    
    @pytest.mark.asyncio
    async def test_parse_large_text_file(self, parser, tmp_path):
        """Test parsing large text file."""
        # Given
        large_content = "Line\n" * 10000  # 10k lines
        large_file = tmp_path / "large.txt"
        large_file.write_text(large_content)
        
        # When
        result = await parser.parse(large_file)
        
        # Then
        assert result.markdown == large_content
        assert result.metadata["file_size"] > 40000  # At least 40KB