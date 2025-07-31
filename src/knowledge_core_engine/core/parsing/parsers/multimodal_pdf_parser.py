import fitz  # PyMuPDF
import base64
from PIL import Image
import io
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..base import BaseParser, ParseResult
from ....utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalPDFParser(BaseParser):
    """多模态PDF解析器，支持提取文本和图像"""
    
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
    
    async def parse(self, file_path: Path) -> ParseResult:
        """解析PDF文件，提取文本和图像"""
        try:
            doc = fitz.open(str(file_path))
            text_chunks = []
            images = []
            
            # 提取内容
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 提取文本
                text = page.get_text()
                if text.strip():
                    self._chunk_text(text, page_num, text_chunks)
                
                # 提取图片
                img_list = page.get_images(full=True)
                for img_index, img in enumerate(img_list):
                    try:
                        base_img = doc.extract_image(img[0])
                        img_data = base_img["image"]
                        images.append({
                            "data": img_data,
                            "page": page_num,
                            "index": img_index
                        })
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
            
            doc.close()
            
            # 构建markdown内容
            markdown_content = self._build_markdown(text_chunks, images)
            
            # 构建元数据
            metadata = {
                "file_name": file_path.name,
                "file_type": "pdf",
                "file_size": file_path.stat().st_size,
                "parse_method": "multimodal_pdf",
                "text_chunks": len(text_chunks),
                "image_count": len(images),
                "encoding": "utf-8"
            }
            
            # 准备图像数据用于多模态处理
            image_data = {
                "text_chunks": text_chunks,
                "images": images
            }
            
            return ParseResult(
                markdown=markdown_content,
                metadata=metadata,
                image=image_data
            )
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise
    
    def _chunk_text(self, text: str, page_num: int, text_chunks: List[Dict]):
        """将长文本分割成适合嵌入的小块"""
        words = text.split()
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            text_chunks.append({
                "content": chunk,
                "page": page_num,
                "type": "text"
            })
    
    def _build_markdown(self, text_chunks: List[Dict], images: List[Dict]) -> str:
        """构建markdown格式的内容"""
        markdown_parts = []
        
        # 按页面组织内容
        pages = {}
        for chunk in text_chunks:
            page_num = chunk["page"]
            if page_num not in pages:
                pages[page_num] = {"text": [], "images": []}
            pages[page_num]["text"].append(chunk["content"])
        
        for img in images:
            page_num = img["page"]
            if page_num not in pages:
                pages[page_num] = {"text": [], "images": []}
            pages[page_num]["images"].append(f"[Image {img['index']} on page {page_num}]")
        
        # 生成markdown
        for page_num in sorted(pages.keys()):
            markdown_parts.append(f"\n## Page {page_num + 1}\n")
            
            # 添加文本内容
            if pages[page_num]["text"]:
                markdown_parts.extend(pages[page_num]["text"])
            
            # 添加图像引用
            if pages[page_num]["images"]:
                markdown_parts.append("\n### Images:")
                markdown_parts.extend(pages[page_num]["images"])
        
        return "\n".join(markdown_parts)