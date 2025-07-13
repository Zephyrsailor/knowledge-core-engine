"""Base parser interface and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class ParseResult:
    """Result of document parsing."""
    
    markdown: str
    metadata: Dict[str, Any]


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