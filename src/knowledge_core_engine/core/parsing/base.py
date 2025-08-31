"""Base parser interface and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List


@dataclass
class ParseResult:
    """Result of document parsing."""
    
    markdown: str
    metadata: Dict[str, Any]
    file_path: str
    file_type: str
    content_list: List[Dict] = None
    md_content: str = "",
    output_dir: str = "",
    execution_time: float = 0.0
    image: List[str] = None
    success: bool = True
    
    @property
    def content(self) -> str:
        """Alias for markdown to maintain compatibility."""
        return self.markdown


class BaseParser(ABC):
    """Abstract base class for all document parsers."""
    
    @abstractmethod
    async def parse(self, file_path: Path) -> ParseResult:
        """
        Parse a document file and return structured result.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ParseResult containing markdown content and metadata
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file type is not supported
            Exception: For other parsing errors
        """
        pass