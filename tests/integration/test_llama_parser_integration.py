"""Integration tests for LlamaParse (requires API key)."""

import pytest
from pathlib import Path
import os

from knowledge_core_engine.core.parsing.llama_parser import LlamaParseWrapper
from knowledge_core_engine.core.parsing.base import ParseResult


@pytest.mark.integration
@pytest.mark.requires_api
class TestLlamaParseIntegration:
    """Integration tests that use real LlamaParse API."""
    
    @pytest.fixture
    def skip_if_no_api_key(self):
        """Skip test if API key is not available."""
        if not os.getenv("LLAMA_CLOUD_API_KEY"):
            pytest.skip("LLAMA_CLOUD_API_KEY not set")
    
    @pytest.mark.asyncio
    async def test_parse_real_text_file(self, skip_if_no_api_key, tmp_path):
        """Test parsing a real text file."""
        # Create test file
        content = """# Test Document
        
This is a test document for integration testing.

## Section 1
- Item 1
- Item 2

## Section 2
This section contains some text."""
        
        test_file = tmp_path / "test.txt"
        test_file.write_text(content)
        
        # Parse with real API
        wrapper = LlamaParseWrapper()
        result = await wrapper.parse(test_file)
        
        # Verify
        assert isinstance(result, ParseResult)
        assert result.markdown  # Should have content
        assert result.metadata["file_name"] == "test.txt"
        assert result.metadata["parse_method"] == "llama_parse"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_parse_real_pdf_file(self, skip_if_no_api_key, tmp_path):
        """Test parsing a real PDF file (if available)."""
        # This test would require a real PDF file
        # For now, we'll create a simple test
        pdf_path = tmp_path / "test.pdf"
        
        # In a real test, you'd have a sample PDF
        # For now, we'll skip if no sample is available
        pytest.skip("Sample PDF not available")
    
    @pytest.mark.asyncio
    async def test_rate_limit_behavior(self, skip_if_no_api_key):
        """Test behavior when approaching rate limits."""
        # This is more of a manual test
        # In production, you'd mock the rate limit response
        pytest.skip("Rate limit testing requires special setup")