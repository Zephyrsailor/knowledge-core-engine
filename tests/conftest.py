"""Shared pytest fixtures and configuration."""

import pytest
import asyncio
from pathlib import Path
import os
from unittest.mock import patch


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "fixtures" / "documents"


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "LLAMA_CLOUD_API_KEY": "llx-test-key-123",
        "DASHSCOPE_API_KEY": "sk-test-dashscope-123",
        "DEEPSEEK_API_KEY": "sk-test-deepseek-123",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing."""
    return """# Sample Document

## Introduction
This is a sample document for testing purposes.

## Main Content
Here is the main content with various elements:

### Subsection 1
- Bullet point 1
- Bullet point 2

### Subsection 2
1. Numbered item 1
2. Numbered item 2

## Tables
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |

## Code Block
```python
def hello_world():
    print("Hello, World!")
```

## Conclusion
This concludes our sample document.
"""


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "file_name": "sample.pdf",
        "file_type": "pdf",
        "file_size": 1024 * 50,  # 50KB
        "parse_method": "llama_parse",
        "parse_time": "2025-01-11T10:00:00Z",
        "page_count": 5,
        "has_tables": True,
        "has_images": False,
        "language": "en",
        "encoding": "utf-8"
    }


@pytest.fixture
async def mock_llama_parse_response():
    """Mock response from LlamaParse API."""
    class MockDocument:
        def __init__(self, text, metadata=None):
            self.text = text
            self.metadata = metadata or {}
    
    return MockDocument