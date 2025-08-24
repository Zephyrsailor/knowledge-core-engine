import base64
import shutil
import uuid
from PIL import Image
import io
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from ..base import BaseParser, ParseResult
from ....utils.logger import get_logger

logger = get_logger(__name__)


class ImageParser(BaseParser):
    """专门处理单独图片上传的解析器"""
    
    # 支持的图片格式
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    
    def __init__(self, knowledge_base_id: str = "defalut", base_upload_dir: str = "data/uploads"):
        """
        初始化图片解析器
        
        Args:
            knowledge_base_id: 知识库ID
            base_upload_dir: 基础上传目录
        """
        self.knowledge_base_id = knowledge_base_id
        self.base_upload_dir = Path(base_upload_dir)
        self.upload_dir = self.base_upload_dir / knowledge_base_id
        
        # 确保上传目录存在
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def parse(self, file_path: Path) -> ParseResult:
        """
        解析图片文件，备份到指定目录并生成元数据
        
        Args:
            file_path: 图片文件路径
            
        Returns:
            ParseResult: 包含图片信息的解析结果
        """
        try:
            # 验证文件存在
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            # 验证文件格式
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {file_path.suffix}")
            
            logger.info(f"Processing image file: {file_path.name}")
            
            # 生成唯一的文件名
            unique_filename = f"{uuid.uuid4().hex}_{file_path.name}"
            backup_path = self.upload_dir / unique_filename
            
            # 备份图片到指定目录
            shutil.copy2(file_path, backup_path)
            logger.info(f"Image backed up to: {backup_path}")
            
            # 读取图片信息
            image_info = self._get_image_info(file_path)
            
            # 读取图片数据用于多模态处理
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # 生成可访问的URL路径
            # 直接使用相对于项目根的路径
            relative_path = Path("data") / "uploads" / self.knowledge_base_id / unique_filename
            accessible_url = f"/{relative_path.as_posix()}"
            
            # 构建markdown内容（简单的图片引用）
            markdown_content = f"![{file_path.stem}]({accessible_url})\n\n图片文件: {file_path.name}"
            
            # 构建元数据
            metadata = {
                "file_name": file_path.name,
                "file_type": "image",
                "file_extension": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
                "parse_method": "image_parser",
                "parse_time": datetime.now(timezone.utc).isoformat(),
                "knowledge_base_id": self.knowledge_base_id,
                "backup_path": str(backup_path),
                "accessible_url": accessible_url,
                "relative_path": str(relative_path),
                "image_width": image_info.get("width"),
                "image_height": image_info.get("height"),
                "image_mode": image_info.get("mode"),
                "image_format": image_info.get("format"),
                "encoding": "binary"
            }
            
            # 准备图像数据用于多模态embedding
            # 修改：保持与multimodal_pdf_parser完全一致的结构
            image_embedding_data = {
                "text_chunks": [],  # 图片没有文本块，保持空列表
                "images": [{
                    "data": image_data,  # 二进制图片数据，用于embedding计算
                    "page": 0,          # 单独图片设为页面0
                    "index": 0          # 图片索引设为0
                }]
            }
            
            return ParseResult(
                markdown=markdown_content,
                metadata=metadata,
                image=image_embedding_data
            )
            
        except Exception as e:
            logger.error(f"Failed to parse image {file_path}: {e}")
            raise
    
    def _get_image_info(self, file_path: Path) -> Dict[str, Any]:
        """
        获取图片的基本信息
        
        Args:
            file_path: 图片文件路径
            
        Returns:
            Dict: 包含图片信息的字典
        """
        try:
            with Image.open(file_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format
                }
        except Exception as e:
            logger.warning(f"Failed to get image info for {file_path}: {e}")
            return {}
    
    def get_backup_directory(self) -> Path:
        """
        获取当前知识库的备份目录
        
        Returns:
            Path: 备份目录路径
        """
        return self.upload_dir
    
    def cleanup_backup(self, backup_path: str) -> bool:
        """
        清理备份文件
        
        Args:
            backup_path: 备份文件路径
            
        Returns:
            bool: 清理是否成功
        """
        try:
            backup_file = Path(backup_path)
            if backup_file.exists():
                backup_file.unlink()
                logger.info(f"Cleaned up backup file: {backup_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cleanup backup file {backup_path}: {e}")
            return False