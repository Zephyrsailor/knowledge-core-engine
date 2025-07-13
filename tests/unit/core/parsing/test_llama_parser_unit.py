"""Unit tests for LlamaParse wrapper (no external dependencies)."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from knowledge_core_engine.core.parsing.llama_parser import LlamaParseWrapper
from knowledge_core_engine.core.parsing.base import ParseResult


@pytest.mark.unit
class TestLlamaParseWrapperUnit:
    """Unit tests for LlamaParseWrapper with mocked dependencies."""
    
    @pytest.fixture
    def mock_llama_parse_class(self):
        """Mock the LlamaParse class itself."""
        with patch('knowledge_core_engine.core.parsing.llama_parser.LlamaParse') as mock_class:
            mock_instance = AsyncMock()
            mock_class.return_value = mock_instance
            yield mock_class, mock_instance
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings to avoid reading .env file."""
        with patch('knowledge_core_engine.core.parsing.llama_parser.get_settings') as mock:
            settings = Mock()
            settings.llama_cloud_api_key = "test-api-key"
            mock.return_value = settings
            yield settings
    
    def test_initialization_with_api_key(self, mock_settings, mock_llama_parse_class):
        """Test wrapper initialization with API key."""
        mock_class, mock_instance = mock_llama_parse_class
        
        # Create wrapper
        wrapper = LlamaParseWrapper(api_key="custom-key")
        
        # Check initialization
        assert wrapper.api_key == "custom-key"
        mock_class.assert_called_once_with(
            api_key="custom-key",
            result_type="markdown",
            verbose=True
        )
    
    def test_initialization_from_settings(self, mock_settings, mock_llama_parse_class):
        """Test wrapper initialization from settings."""
        mock_class, mock_instance = mock_llama_parse_class
        
        # Create wrapper without explicit API key
        wrapper = LlamaParseWrapper()
        
        # Should use key from settings
        assert wrapper.api_key == "test-api-key"
    
    @pytest.mark.asyncio
    async def test_parse_success(self, mock_settings, mock_llama_parse_class, tmp_path):
        """Test successful parsing."""
        mock_class, mock_instance = mock_llama_parse_class
        
        # Setup mock response
        mock_doc = MagicMock()
        mock_doc.text = "# Parsed Content\n\nThis is the content."
        mock_doc.metadata = {"pages": 5, "has_tables": True}
        mock_instance.aload_data.return_value = [mock_doc]
        
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")
        
        # Parse
        wrapper = LlamaParseWrapper()
        result = await wrapper.parse(test_file)
        
        # Verify
        assert isinstance(result, ParseResult)
        assert "# Parsed Content" in result.markdown
        assert result.metadata["file_name"] == "test.pdf"
        assert result.metadata["pages"] == 5
        assert result.metadata["has_tables"] is True
        mock_instance.aload_data.assert_called_once_with(str(test_file))
    
    @pytest.mark.asyncio
    async def test_parse_multiple_documents(self, mock_settings, mock_llama_parse_class, tmp_path):
        """Test parsing when LlamaParse returns multiple document objects."""
        mock_class, mock_instance = mock_llama_parse_class
        
        # Setup mock response with multiple docs
        docs = []
        for i in range(3):
            doc = MagicMock()
            doc.text = f"Page {i+1} content"
            doc.metadata = {"page": i+1}
            docs.append(doc)
        mock_instance.aload_data.return_value = docs
        
        # Create test file
        test_file = tmp_path / "multi.pdf"
        test_file.write_bytes(b"PDF")
        
        # Parse
        wrapper = LlamaParseWrapper()
        result = await wrapper.parse(test_file)
        
        # Verify content is combined
        assert "Page 1 content" in result.markdown
        assert "Page 2 content" in result.markdown
        assert "Page 3 content" in result.markdown
        assert result.metadata["page"] == 3  # Last page metadata
    
    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self, mock_settings, mock_llama_parse_class):
        """Test parsing non-existent file."""
        wrapper = LlamaParseWrapper()
        
        with pytest.raises(FileNotFoundError):
            await wrapper.parse(Path("does_not_exist.pdf"))
    
    @pytest.mark.asyncio
    async def test_parse_api_error(self, mock_settings, mock_llama_parse_class, tmp_path):
        """Test handling of API errors."""
        mock_class, mock_instance = mock_llama_parse_class
        
        # Setup mock to raise error
        mock_instance.aload_data.side_effect = Exception("API Error: Rate limit exceeded")
        
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF")
        
        # Parse should propagate error
        wrapper = LlamaParseWrapper()
        with pytest.raises(Exception, match="API Error: Rate limit exceeded"):
            await wrapper.parse(test_file)
    
    def test_custom_parsing_options(self, mock_settings, mock_llama_parse_class):
        """Test initialization with custom parsing options."""
        mock_class, mock_instance = mock_llama_parse_class
        
        # Create wrapper with custom options
        wrapper = LlamaParseWrapper(
            result_type="text",
            parsing_instruction="Extract tables only",
            skip_diagonal_text=True,
            language="zh"
        )
        
        # Verify client was initialized with custom options
        mock_class.assert_called_once()
        call_kwargs = mock_class.call_args[1]
        assert call_kwargs["result_type"] == "text"
        assert call_kwargs["parsing_instruction"] == "Extract tables only"
        assert call_kwargs["skip_diagonal_text"] is True
        assert call_kwargs["language"] == "zh"
    
    def test_missing_api_key_error(self):
        """Test error when API key is missing."""
        with patch('knowledge_core_engine.core.parsing.llama_parser.get_settings') as mock_settings:
            # Mock settings with no API key
            settings = Mock()
            settings.llama_cloud_api_key = None
            mock_settings.return_value = settings
            
            with patch.dict('os.environ', {}, clear=True):
                with pytest.raises(ValueError, match="LLAMA_CLOUD_API_KEY"):
                    LlamaParseWrapper()