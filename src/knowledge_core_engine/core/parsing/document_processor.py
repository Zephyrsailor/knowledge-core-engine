"""Document processor that orchestrates parsing with different parsers."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Set

from knowledge_core_engine.core.parsing.base import BaseParser, ParseResult
from knowledge_core_engine.core.parsing.llama_parser import LlamaParseWrapper
from knowledge_core_engine.core.parsing.parsers import TextParser, MarkdownParser
from knowledge_core_engine.utils.config import get_settings
from knowledge_core_engine.utils.logger import get_logger, log_detailed, log_step

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Document processor that routes files to appropriate parsers.
    
    This class acts as a facade for the parsing system, handling:
    - Parser selection based on file type
    - Caching of parsed results
    - Error handling and logging
    """
    
    def __init__(
        self,
        cache_enabled: Optional[bool] = None,
        cache_dir: Optional[Path] = None,
        **llama_parse_kwargs
    ):
        """
        Initialize document processor.
        
        Args:
            cache_enabled: Whether to enable caching
            cache_dir: Directory for cache storage
            **llama_parse_kwargs: Additional arguments for LlamaParse
        """
        settings = get_settings()
        
        self.cache_enabled = cache_enabled if cache_enabled is not None else settings.enable_cache
        self.cache_dir = cache_dir or settings.cache_dir
        
        # Ensure cache directory exists
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parsers
        self._init_parsers(**llama_parse_kwargs)
        
        logger.info("DocumentProcessor initialized")
    
    def _init_parsers(self, **llama_parse_kwargs):
        """Initialize all available parsers."""
        # Simple parsers
        self._text_parser = TextParser()
        self._markdown_parser = MarkdownParser()
        
        # LlamaParse for complex documents
        self._llama_parser = LlamaParseWrapper(**llama_parse_kwargs)
        
        # Parser mapping by file extension
        self._parsers: Dict[str, BaseParser] = {
            '.txt': self._text_parser,
            '.md': self._markdown_parser,
            # LlamaParse handles these formats
            '.pdf': self._llama_parser,
            '.docx': self._llama_parser,
            '.doc': self._llama_parser,
            '.pptx': self._llama_parser,
            '.ppt': self._llama_parser,
            '.xlsx': self._llama_parser,
            '.xls': self._llama_parser,
            '.csv': self._llama_parser,
        }
    
    @property
    def supported_extensions(self) -> Set[str]:
        """Get set of supported file extensions."""
        return set(self._parsers.keys())
    
    def register_parser(self, extension: str, parser: BaseParser):
        """
        Register a custom parser for a file extension.
        
        Args:
            extension: File extension (e.g., '.custom')
            parser: Parser instance that implements BaseParser
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        self._parsers[extension] = parser
        logger.info(f"Registered parser for {extension}")
    
    @log_step("Document Processing")
    async def process(self, file_path: Path) -> ParseResult:
        """
        Process a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ParseResult containing markdown and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file type is supported
        extension = file_path.suffix.lower()
        if extension not in self._parsers:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {', '.join(sorted(self.supported_extensions))}"
            )
        
        file_size = file_path.stat().st_size
        log_detailed(f"Processing file: {file_path.name}", 
                    data={"type": extension, "size": file_size})
        
        # Check cache first
        if self.cache_enabled:
            cached_result = await self._get_from_cache(file_path)
            if cached_result:
                logger.debug(f"Using cached parse result for {file_path.name}")
                return cached_result
        
        # Get appropriate parser
        parser = self._parsers[extension]
        
        # Parse document
        logger.info(f"Parsing {file_path.name} with {parser.__class__.__name__}")
        
        result = await parser.parse(file_path)
        
        # Add file path to metadata
        result.metadata['file_path'] = str(file_path.absolute())
        
        # Save to cache
        if self.cache_enabled:
            await self._save_to_cache(file_path, result)
        
        return result
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for file."""
        # Use file path, size, and modification time for cache key
        stat = file_path.stat()
        key_data = f"{file_path.absolute()}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def _get_from_cache(self, file_path: Path) -> Optional[ParseResult]:
        """Get parsed result from cache."""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with cache_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            return ParseResult(
                markdown=data['markdown'],
                metadata=data['metadata']
            )
        except Exception as e:
            logger.warning(f"Error reading cache for {file_path.name}: {e}")
            return None
    
    async def _save_to_cache(self, file_path: Path, result: ParseResult) -> None:
        """Save parsed result to cache."""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                'markdown': result.markdown,
                'metadata': result.metadata,
                'cached_at': result.metadata.get('parse_time', '')
            }
            
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Cached result for {file_path.name}")
        except Exception as e:
            logger.warning(f"Error saving cache for {file_path.name}: {e}")