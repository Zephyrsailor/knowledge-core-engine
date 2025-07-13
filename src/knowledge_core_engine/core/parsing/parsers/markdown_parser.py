"""Markdown file parser."""

from pathlib import Path
from datetime import datetime, timezone

from knowledge_core_engine.core.parsing.base import BaseParser, ParseResult
from knowledge_core_engine.utils.logger import setup_logger

logger = setup_logger(__name__)


class MarkdownParser(BaseParser):
    """Parser for markdown files (passthrough)."""
    
    async def parse(self, file_path: Path) -> ParseResult:
        """
        Parse a markdown file (passthrough).
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            ParseResult with markdown content and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Parsing markdown file: {file_path.name}")
        
        # Read file content (passthrough for markdown)
        content = file_path.read_text(encoding='utf-8')
        
        # Build metadata
        metadata = {
            "file_name": file_path.name,
            "file_type": "md",
            "file_size": file_path.stat().st_size,
            "parse_method": "markdown_parser",
            "parse_time": datetime.now(timezone.utc).isoformat(),
            "encoding": "utf-8"
        }
        
        return ParseResult(markdown=content, metadata=metadata)