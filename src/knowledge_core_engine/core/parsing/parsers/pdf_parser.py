"""Plain pdf file parser."""
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime, timezone

from knowledge_core_engine.core.parsing.base import BaseParser, ParseResult
from knowledge_core_engine.utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFParser(BaseParser):
    """Parser for plain pdf files."""

    async def parse(self, file_path: Path) -> ParseResult:

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Parsing PDF file: {file_path.name}")

        text_array = []
        image_array = []
        # Read file content
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # 提取文本
            text_array.append(page.get_text())
            # 提取图片
            img_list = page.get_images(full=True)
            for img_index, img in enumerate(img_list):
                base_img = doc.extract_image(img[0])
                img_data = base_img["image"]
                image_array.extend({
                    "data": img_data,
                    "page": page_num,
                    "index": img_index
                })

        # Build metadata
        metadata = {
            "file_name": file_path.name,
            "file_type": "pdf",
            "file_size": file_path.stat().st_size,
            "parse_method": "pdf_parser",
            "parse_time": datetime.now(timezone.utc).isoformat(),
            "encoding": "utf-8"
        }
        content =  "\n\n".join(text_array)
        return ParseResult(markdown=content, metadata=metadata,image=image_array)