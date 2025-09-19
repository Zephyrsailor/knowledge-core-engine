"""MinerU-based parser for advanced multimodal document processing."""
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from knowledge_core_engine.core.parsing.base import BaseParser, ParseResult
from knowledge_core_engine.core.parsing.utils.mineru_utils import MineruUtils
from knowledge_core_engine.utils.logger import get_logger

logger = get_logger(__name__)


class MineruParser(BaseParser):
    """Parser using MinerU for advanced multimodal document processing.
    
    Supports:
    - PDF documents
    - Images (JPG, PNG, BMP, TIFF, GIF, WebP, SVG)
    - Office documents (DOC, DOCX, PPT, PPTX, XLS, XLSX)
    - Text files (TXT, MD)
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        # PDF
        '.pdf',
        # Images
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.svg',
        # Office documents
        '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
        # Text files
        '.txt', '.md'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MinerU parser.
        
        Args:
            config: Configuration for MinerU parsing
        """
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for MinerU parsing."""
        return {
            "method": "auto",
            "lang": "ch",
            "backend": "pipeline",
            "source": "local",
            "formula": True,
            "table": True,
            "device": None,
            "output_dir": None
        }
    
    async def parse(self, file_path: Path) -> ParseResult:
        """Parse document using MinerU.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ParseResult containing markdown content and metadata
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file type is not supported
            Exception: For other parsing errors
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_suffix = file_path.suffix.lower()
        if file_suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_suffix}. Supported types: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}")
        
        logger.info(f"Parsing {file_path.name} with MinerU")
        
        try:
            start_time = time.time()
            # Prepare output directory
            output_dir = self.config.get("OUTPUT_DIR")
            if not output_dir:
                output_dir = file_path.parent / "parse_output" / file_path.stem
            else:
                output_dir = Path(output_dir)
            
            # Determine file type and parse accordingly
            if file_suffix == '.pdf':
                content_list, md_content = await self._parse_pdf(file_path, output_dir)
                file_type = "pdf"
            elif file_suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp', '.svg'}:
                content_list, md_content = await self._parse_image(file_path, output_dir)
                file_type = "image"
            elif file_suffix in {'.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}:
                content_list, md_content = await self._parse_office_doc(file_path, output_dir)
                file_type = "office"
            elif file_suffix in {'.txt', '.md'}:
                content_list, md_content = await self._parse_text_file(file_path, output_dir)
                file_type = "text"
            else:
                raise ValueError(f"Unsupported file type: {file_suffix}")

            metadata = {
                "file_name": file_path.name,
                "file_type": "pdf",
                "file_size": file_path.stat().st_size,
                "parse_method": "mineru",
                "parse_time": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(output_dir),
                "content_list_count": len(content_list) if content_list else 0,
                "mineru_config": self.config
            }

            logger.info(f"Successfully parsed {file_path.name} with MinerU")

            execution_time = time.time() - start_time
            
            return ParseResult(
                success = True,
                file_path = str(file_path),
                file_type = file_type,
                content_list = content_list,
                md_content = md_content,
                output_dir = str(output_dir),
                execution_time = execution_time,
                markdown=md_content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing {file_path} with MinerU: {str(e)}")
            raise

    async def _parse_pdf(self, file_path: Path, output_dir: Path) -> tuple:
        """Parse PDF file."""
        return MineruUtils.parse_pdf(
            pdf_path=file_path,
            output_dir=output_dir,
            method=self.config.get("method", "auto"),
            lang=self.config.get("lang", "ch"),
            backend=self.config.get("backend", "pipeline"),
            source=self.config.get("source", "local"),
            formula=self.config.get("formula", True),
            table=self.config.get("table", True),
            device=self.config.get("device")
        )
    
    async def _parse_image(self, file_path: Path, output_dir: Path) -> tuple:
        """Parse image file."""
        return MineruUtils.parse_image(
            image_path=file_path,
            output_dir=output_dir,
            method=self.config.get("method", "auto"),
            lang=self.config.get("lang", "ch"),
            backend=self.config.get("backend", "pipeline"),
            source=self.config.get("source", "local"),
            device=self.config.get("device")
        )
    
    async def _parse_office_doc(self, file_path: Path, output_dir: Path) -> tuple:
        """Parse Office document."""
        return MineruUtils.parse_office_doc(
            doc_path=file_path,
            output_dir=output_dir,
            method=self.config.get("method", "auto"),
            lang=self.config.get("lang", "ch"),
            backend=self.config.get("backend", "pipeline"),
            source=self.config.get("source", "local"),
            formula=self.config.get("formula", True),
            table=self.config.get("table", True),
            device=self.config.get("device")
        )
    
    async def _parse_text_file(self, file_path: Path, output_dir: Path) -> tuple:
        """Parse text file."""
        return MineruUtils.parse_text_file(
            text_path=file_path,
            output_dir=output_dir,
            method=self.config.get("method", "auto"),
            lang=self.config.get("lang", "ch"),
            backend=self.config.get("backend", "pipeline"),
            source=self.config.get("source", "local"),
            formula=self.config.get("formula", True),
            table=self.config.get("table", True),
            device=self.config.get("device")
        )
