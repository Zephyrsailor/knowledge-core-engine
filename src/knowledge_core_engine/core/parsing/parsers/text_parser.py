"""Plain text file parser."""

from pathlib import Path
from datetime import datetime, timezone

from knowledge_core_engine.core.parsing.base import BaseParser, ParseResult
from knowledge_core_engine.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextParser(BaseParser):
    """Parser for plain text files."""
    
    async def parse(self, file_path: Path) -> ParseResult:
        """
        Parse a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            ParseResult with text content and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Parsing text file: {file_path.name}")
        
        # Read file content
        content = file_path.read_text(encoding='utf-8')
        
        # Build metadata
        metadata = {
            "file_name": file_path.name,
            "file_type": "txt",
            "file_size": file_path.stat().st_size,
            "parse_method": "text_parser",
            "parse_time": datetime.now(timezone.utc).isoformat(),
            "encoding": "utf-8"
        }
        
        return ParseResult(markdown=content, metadata=metadata)