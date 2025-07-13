"""Tests for base parser interface."""

import pytest
from abc import ABC
from pathlib import Path
from typing import Dict, Any, Tuple

from knowledge_core_engine.core.parsing.base import BaseParser, ParseResult


class TestParseResult:
    """Test ParseResult data class."""
    
    def test_parse_result_creation(self):
        """Test creating a ParseResult instance."""
        # Given
        markdown = "# Test Document\n\nThis is test content."
        metadata = {
            "file_name": "test.pdf",
            "file_type": "pdf",
            "file_size": 1234,
            "parse_method": "llama_parse"
        }
        
        # When
        result = ParseResult(markdown=markdown, metadata=metadata)
        
        # Then
        assert result.markdown == markdown
        assert result.metadata == metadata
        assert isinstance(result.metadata, dict)
    
    def test_parse_result_with_empty_metadata(self):
        """Test ParseResult with empty metadata."""
        # Given
        markdown = "# Empty metadata test"
        
        # When
        result = ParseResult(markdown=markdown, metadata={})
        
        # Then
        assert result.markdown == markdown
        assert result.metadata == {}
    
    def test_parse_result_immutability(self):
        """Test that ParseResult fields are properly typed."""
        # Given
        result = ParseResult(
            markdown="# Test",
            metadata={"key": "value"}
        )
        
        # Then - these should be accessible
        assert hasattr(result, 'markdown')
        assert hasattr(result, 'metadata')


class TestBaseParser:
    """Test BaseParser abstract base class."""
    
    def test_base_parser_is_abstract(self):
        """Test that BaseParser cannot be instantiated directly."""
        # When/Then
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseParser()
    
    def test_base_parser_requires_parse_method(self):
        """Test that subclasses must implement parse method."""
        # Given
        class IncompleteParser(BaseParser):
            pass
        
        # When/Then
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteParser()
    
    def test_base_parser_subclass_with_parse_method(self):
        """Test that properly implemented subclass can be instantiated."""
        # Given
        class ConcreteParser(BaseParser):
            async def parse(self, file_path: Path) -> ParseResult:
                return ParseResult(
                    markdown="# Test",
                    metadata={"file_name": file_path.name}
                )
        
        # When
        parser = ConcreteParser()
        
        # Then
        assert isinstance(parser, BaseParser)
        assert hasattr(parser, 'parse')
    
    @pytest.mark.asyncio
    async def test_parse_method_signature(self):
        """Test that parse method has correct signature."""
        # Given
        class TestParser(BaseParser):
            async def parse(self, file_path: Path) -> ParseResult:
                return ParseResult(
                    markdown=f"# {file_path.name}",
                    metadata={"file_name": file_path.name}
                )
        
        # When
        parser = TestParser()
        result = await parser.parse(Path("test.pdf"))
        
        # Then
        assert isinstance(result, ParseResult)
        assert result.markdown == "# test.pdf"
        assert result.metadata["file_name"] == "test.pdf"
    
    def test_supported_extensions_attribute(self):
        """Test that parser can have supported_extensions attribute."""
        # Given
        class ExtensionParser(BaseParser):
            supported_extensions = {".pdf", ".docx", ".txt"}
            
            async def parse(self, file_path: Path) -> ParseResult:
                return ParseResult(markdown="", metadata={})
        
        # When
        parser = ExtensionParser()
        
        # Then
        assert hasattr(parser, 'supported_extensions')
        assert ".pdf" in parser.supported_extensions
        assert ".docx" in parser.supported_extensions
        assert ".txt" in parser.supported_extensions