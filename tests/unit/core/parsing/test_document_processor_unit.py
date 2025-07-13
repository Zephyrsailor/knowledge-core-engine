"""Unit tests for document processor (fully mocked)."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

from knowledge_core_engine.core.parsing.document_processor import DocumentProcessor
from knowledge_core_engine.core.parsing.base import ParseResult


@pytest.mark.unit
class TestDocumentProcessorUnit:
    """Unit tests for DocumentProcessor with mocked dependencies."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings to avoid reading .env."""
        with patch('knowledge_core_engine.core.parsing.document_processor.get_settings') as mock:
            settings = Mock()
            settings.enable_cache = False
            settings.cache_dir = Path("/tmp/cache")
            settings.llama_cloud_api_key = "test-key"
            mock.return_value = settings
            yield settings
    
    @pytest.fixture
    def mock_llama_parser_class(self):
        """Mock LlamaParseWrapper class."""
        with patch('knowledge_core_engine.core.parsing.document_processor.LlamaParseWrapper') as mock_class:
            mock_instance = Mock()
            mock_instance.parse = AsyncMock()
            mock_class.return_value = mock_instance
            yield mock_class, mock_instance
    
    @pytest.fixture
    def processor(self, mock_settings, mock_llama_parser_class):
        """Create processor with mocked dependencies."""
        return DocumentProcessor()
    
    @pytest.mark.asyncio
    async def test_process_pdf_with_llama_parser(self, processor, mock_llama_parser_class, tmp_path):
        """Test processing PDF uses LlamaParse."""
        _, mock_parser = mock_llama_parser_class
        
        # Setup
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"PDF content")
        
        expected_result = ParseResult(
            markdown="# Parsed PDF\nContent here",
            metadata={"file_type": "pdf", "source": "llama_parse"}
        )
        mock_parser.parse.return_value = expected_result
        
        # Execute
        result = await processor.process(pdf_file)
        
        # Verify
        assert result == expected_result
        mock_parser.parse.assert_called_once_with(pdf_file)
    
    @pytest.mark.asyncio
    async def test_process_text_file(self, processor, tmp_path):
        """Test processing text file uses TextParser."""
        # Setup
        text_file = tmp_path / "test.txt"
        text_content = "Plain text content"
        text_file.write_text(text_content)
        
        # Execute
        result = await processor.process(text_file)
        
        # Verify
        assert result.markdown == text_content
        assert result.metadata["file_type"] == "txt"
        assert result.metadata["parse_method"] == "text_parser"
    
    @pytest.mark.asyncio
    async def test_process_markdown_file(self, processor, tmp_path):
        """Test processing markdown file."""
        # Setup
        md_file = tmp_path / "test.md"
        md_content = "# Markdown\n\nContent"
        md_file.write_text(md_content)
        
        # Execute
        result = await processor.process(md_file)
        
        # Verify
        assert result.markdown == md_content
        assert result.metadata["file_type"] == "md"
        assert result.metadata["parse_method"] == "markdown_parser"
    
    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, processor, tmp_path):
        """Test error for unsupported file type."""
        # Setup
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("data")
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Unsupported file type"):
            await processor.process(unsupported)
    
    @pytest.mark.asyncio
    async def test_nonexistent_file(self, processor):
        """Test error for non-existent file."""
        # Execute & Verify
        with pytest.raises(FileNotFoundError):
            await processor.process(Path("does_not_exist.pdf"))
    
    def test_supported_extensions(self, processor):
        """Test supported extensions property."""
        extensions = processor.supported_extensions
        
        assert '.pdf' in extensions
        assert '.docx' in extensions
        assert '.txt' in extensions
        assert '.md' in extensions
    
    def test_register_custom_parser(self, processor):
        """Test registering custom parser."""
        # Setup
        custom_parser = Mock()
        custom_parser.parse = AsyncMock(
            return_value=ParseResult("custom", {"type": "custom"})
        )
        
        # Register
        processor.register_parser('.custom', custom_parser)
        
        # Verify
        assert '.custom' in processor.supported_extensions
        assert processor._parsers['.custom'] == custom_parser
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self, processor, tmp_path):
        """Test processing with cache disabled."""
        # Ensure cache is disabled
        processor.cache_enabled = False
        
        # Create file
        text_file = tmp_path / "test.txt"
        text_file.write_text("Content")
        
        # Process twice
        result1 = await processor.process(text_file)
        result2 = await processor.process(text_file)
        
        # Both should be processed (no caching)
        assert result1.markdown == result2.markdown
    
    @pytest.mark.asyncio
    async def test_cache_enabled(self, tmp_path):
        """Test processing with cache enabled."""
        # Create processor with cache enabled
        with patch('knowledge_core_engine.core.parsing.document_processor.get_settings') as mock_settings:
            settings = Mock()
            settings.enable_cache = True
            settings.cache_dir = tmp_path / "cache"
            settings.llama_cloud_api_key = "test-key"
            mock_settings.return_value = settings
            
            processor = DocumentProcessor()
            
            # Create file
            text_file = tmp_path / "test.txt"
            text_file.write_text("Content")
            
            # Process twice
            result1 = await processor.process(text_file)
            
            # Verify cache file was created
            cache_files = list((tmp_path / "cache").glob("*.json"))
            assert len(cache_files) == 1
            
            # Second call should use cache
            result2 = await processor.process(text_file)
            assert result1.markdown == result2.markdown
    
    @pytest.mark.asyncio
    async def test_error_propagation(self, processor, mock_llama_parser_class, tmp_path):
        """Test that parser errors are propagated."""
        _, mock_parser = mock_llama_parser_class
        
        # Setup error
        mock_parser.parse.side_effect = Exception("Parse error")
        
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"PDF")
        
        # Execute & Verify
        with pytest.raises(Exception, match="Parse error"):
            await processor.process(pdf_file)