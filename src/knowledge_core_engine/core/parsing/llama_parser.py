"""LlamaParse integration for document parsing."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from llama_parse import LlamaParse

from knowledge_core_engine.core.parsing.base import BaseParser, ParseResult
from knowledge_core_engine.utils.config import get_settings
from knowledge_core_engine.utils.logger import setup_logger

logger = setup_logger(__name__)


class LlamaParseWrapper(BaseParser):
    """Wrapper for LlamaParse API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        result_type: str = "markdown",
        parsing_instruction: Optional[str] = None,
        skip_diagonal_text: bool = False,
        invalidate_cache: bool = False,
        verbose: bool = True,
        language: str = "en",
    ):
        """
        Initialize LlamaParse wrapper.
        
        Args:
            api_key: LlamaParse API key (uses env var if not provided)
            result_type: Output format ("markdown" or "text")
            parsing_instruction: Custom parsing instructions
            skip_diagonal_text: Whether to skip diagonal text
            invalidate_cache: Whether to invalidate cache
            verbose: Whether to show verbose output
            language: Document language
        """
        settings = get_settings()
        # Use settings (which uses KCE_ prefix) or direct KCE_ env var
        self.api_key = api_key or settings.llama_cloud_api_key or os.getenv("KCE_LLAMA_CLOUD_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "LLAMA_CLOUD_API_KEY not found. Please set KCE_LLAMA_CLOUD_API_KEY in .env or environment variables."
            )
        
        self.result_type = result_type
        self.parsing_instruction = parsing_instruction
        self.skip_diagonal_text = skip_diagonal_text
        self.invalidate_cache = invalidate_cache
        self.verbose = verbose
        self.language = language
        
        # Initialize LlamaParse client
        self._init_client()
    
    def _init_client(self):
        """Initialize LlamaParse client with configuration."""
        client_kwargs = {
            "api_key": self.api_key,
            "result_type": self.result_type,
            "verbose": self.verbose,
        }
        
        if self.parsing_instruction:
            client_kwargs["parsing_instruction"] = self.parsing_instruction
        
        if self.skip_diagonal_text:
            client_kwargs["skip_diagonal_text"] = self.skip_diagonal_text
            
        if self.invalidate_cache:
            client_kwargs["invalidate_cache"] = self.invalidate_cache
            
        if self.language != "en":
            client_kwargs["language"] = self.language
        
        self.client = LlamaParse(**client_kwargs)
    
    async def parse(self, file_path: Path) -> ParseResult:
        """
        Parse document using LlamaParse.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ParseResult with markdown content and metadata
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Parsing document with LlamaParse: {file_path}")
        
        try:
            # Parse document
            documents = await self.client.aload_data(str(file_path))
            
            # Combine text from all document objects
            markdown_parts = []
            combined_metadata = {
                "file_name": file_path.name,
                "file_type": file_path.suffix.lstrip("."),
                "file_size": file_path.stat().st_size,
                "parse_method": "llama_parse",
                "parse_time": datetime.now(timezone.utc).isoformat(),
            }
            
            for doc in documents:
                markdown_parts.append(doc.text)
                
                # Merge metadata from document if available
                if hasattr(doc, 'metadata') and doc.metadata:
                    combined_metadata.update(doc.metadata)
            
            # Join all parts
            markdown = "\n\n".join(markdown_parts)
            
            logger.info(f"Successfully parsed {file_path.name}")
            
            return ParseResult(
                markdown=markdown,
                metadata=combined_metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            raise