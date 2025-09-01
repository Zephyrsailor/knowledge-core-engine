#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态RAG系统的文档切片代理

该模块实现了对MinerU解析结果的智能切片功能，支持文本、图片、表格等多种数据类型的切片策略。
切片结果将用于后续的向量化、存储与检索。

Author: fanjs
Date: 2025-08-27
"""

import os
import re
import json
import base64
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """切片类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    MIXED = "mixed"  # 混合类型切片

class ChunkStrategy(Enum):
    """切片策略枚举"""
    FIXED_SIZE = "fixed_size"  # 固定大小切片
    SEMANTIC = "semantic"  # 语义边界切片
    SENTENCE = "sentence"  # 句子边界切片
    PARAGRAPH = "paragraph"  # 段落边界切片
    DOCUMENT_STRUCTURE = "document_structure"  # 文档结构切片
    HYBRID = "hybrid"  # 混合策略

@dataclass
class ChunkConfig:
    """切片配置类"""
    max_chunk_size: int = 1000
    min_chunk_size: int = 100
    overlap_size: int = 100
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    include_page_info: bool = True
    include_position_info: bool = True
    include_source_info: bool = True

    # 策略配置
    text_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC
    table_strategy: ChunkStrategy = ChunkStrategy.DOCUMENT_STRUCTURE
    image_strategy: ChunkStrategy = ChunkStrategy.DOCUMENT_STRUCTURE

    # 图片切片特定配置
    image_as_single_chunk: bool = True  # 图片作为单个切片
    include_image_caption: bool = True  # 包含图片标题
    include_image_context: bool = True  # 包含图片上下文

    # 表格切片特定配置
    table_as_single_chunk: bool = True  # 表格作为单个切片
    include_table_caption: bool = True  # 包含表格标题
    include_table_context: bool = True  # 包含表格上下文


@dataclass
class Chunk:
    """切片数据类"""
    chunk_id: str
    chunk_type: ChunkType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_idx: Optional[int] = None
    chunk_idx: int = 0
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    source_file: Optional[str] = None
    parent_document: Optional[str] = None
    content_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            'chunk_id': self.chunk_id,
            'chunk_type': self.chunk_type,
            'content': self.content,
            'metadata': self.metadata,
            'page_idx': self.page_idx,
            'chunk_idx': self.chunk_idx,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'source_file': self.source_file,
            'parent_document': self.parent_document,
            'content_path': self.content_path
        }
        
        # 将embedding_type提升到顶层
        if 'embedding_type' in self.metadata:
            result['embedding_type'] = self.metadata['embedding_type']
        
        return result


class ChunkAgent:
    """文档切片代理类
    
    负责对MinerU解析的文档结果进行智能切片，支持多种数据类型和切片策略。
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """初始化切片代理
        
        Args:
            config: 切片配置，如果为None则使用默认配置
        """
        self.config = config or ChunkConfig()
        self.chunks: List[Chunk] = []
        self.chunk_counter = 0

        self.sentence_pattern = re.compile(r'[.!?。！？]+\s*')
        # 段落分割正则表达式
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        logger.info(f"ChunkAgent initialized with config: {self.config}")
    
    def chunk_document(self, 
                      content_list_path: str, 
                      output_dir: str,
                      document_name: Optional[str] = None) -> List[Chunk]:
        """对文档进行切片处理
        
        Args:
            content_list_path: MinerU输出的content_list.json文件路径
            output_dir: 输出目录路径
            document_name: 文档名称，用于生成chunk_id
            
        Returns:
            切片结果列表
        """
        try:
            # 读取content_list.json
            with open(content_list_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            if document_name is None:
                document_name = Path(content_list_path).parent.parent.name
            
            logger.info(f"开始处理文档: {document_name}, 共{len(content_list)}个元素")
            
            # 重置切片计数器
            self.chunk_counter = 0
            self.chunks = []

            # 根据数据类型分组处理
            grouped_content = self._group_content_by_type(content_list)

            # 处理每个内容组
            for group in grouped_content:
                chunks = self._process_content_group(group, document_name, output_dir)
                self.chunks.extend(chunks)
            
            # 处理每个内容元素
            # for i, content_item in enumerate(content_list):
            #     chunk = self._process_content_item(content_item, document_name, output_dir, i)
            #     if chunk:
            #         self.chunks.append(chunk)

            # 后处理：添加关联信息和优化切片
            self._post_process_chunks()
            
            logger.info(f"文档切片完成，共生成{len(self.chunks)}个切片")
            return self.chunks
            
        except Exception as e:
            logger.error(f"文档切片处理失败: {str(e)}")
            raise
    
    def _process_content_item(self, content_item: Dict, document_name: str, output_dir: str, index: int) -> Optional[Chunk]:
        """处理单个内容项"""
        content_type = content_item.get('type', 'text')
        
        if content_type == 'text':
            return self._process_text_content(content_item, document_name, index)
        elif content_type == 'image':
            return self._process_image_content(content_item, document_name, output_dir, index)
        elif content_type == 'table':
            return self._process_table_content(content_item, document_name, index)
        else:
            logger.warning(f"未知的内容类型: {content_type}")
            return None
    
    def _process_text_content(self, content_item: Dict, document_name: str, index: int) -> Chunk:
        """处理文本内容"""
        text = content_item.get('text', '')
        
        chunk = Chunk(
            chunk_id=f"{document_name}_text_{index}",
            chunk_type='text',
            content=text,
            metadata={
                'text_level': content_item.get('text_level', 0),
                'source_document': document_name
            },
            page_idx=content_item.get('page_idx', 0),
            chunk_idx=index,
            source_file=document_name,
            parent_document=document_name
        )
        
        return chunk
    
    def _process_image_content(self, content_item: Dict, document_name: str, output_dir: str, index: int) -> Chunk:
        """处理图片内容"""
        img_path = content_item.get('img_path', '')
        image_caption = content_item.get('image_caption', [])
        
        # 构建图片描述文本
        caption_text = '\n'.join(image_caption) if image_caption else ''
        content_text = f"图片: {img_path}\n描述: {caption_text}" if caption_text else f"图片: {img_path}"
        
        chunk = Chunk(
            chunk_id=f"{document_name}_image_{index}",
            chunk_type='image',
            content=content_text,
            metadata={
                'image_path': img_path,
                'has_caption': bool(image_caption),
                'source_document': document_name
            },
            page_idx=content_item.get('page_idx', 0),
            chunk_idx=index,
            source_file=document_name,
            parent_document=document_name,
            content_path=img_path
        )
        
        return chunk
    
    def _process_table_content(self, content_item: Dict, document_name: str, index: int) -> Chunk:
        """处理表格内容"""
        # 这里可以根据实际的表格数据结构进行处理
        table_content = str(content_item)  # 简化处理
        
        chunk = Chunk(
            chunk_id=f"{document_name}_table_{index}",
            chunk_type='table',
            content=table_content,
            metadata={
                'source_document': document_name
            },
            page_idx=content_item.get('page_idx', 0),
            chunk_idx=index,
            source_file=document_name,
            parent_document=document_name
        )
        
        return chunk

    def _group_content_by_type(self, content_list: List[Dict]) -> List[List[Dict]]:
        """根据内容类型对数据进行分组

        将连续的同类型内容分组，以便应用相应的切片策略。

        Args:
            content_list: MinerU解析的内容列表

        Returns:
            分组后的内容列表
        """
        if not content_list:
            return []

        groups = []
        current_group = [content_list[0]]
        current_type = content_list[0].get('type')

        for item in content_list[1:]:
            item_type = item.get('type')

            # 如果类型相同且都是文本，继续添加到当前组
            if (item_type == current_type and item_type == 'text' and
                    len(current_group) < 10):  # 限制文本组大小，避免过大
                current_group.append(item)
            else:
                # 开始新组
                groups.append(current_group)
                current_group = [item]
                current_type = item_type

        # 添加最后一组
        if current_group:
            groups.append(current_group)

        return groups

    def _process_content_group(self,
                               group: List[Dict],
                               document_name: str,
                               output_dir: str) -> List[Chunk]:
        """处理内容组

        Args:
            group: 内容组
            document_name: 文档名称
            output_dir: 输出目录

        Returns:
            切片列表
        """
        if not group:
            return []

        group_type = group[0].get('type')

        if group_type == 'text':
            return self._chunk_text_group(group, document_name)
        elif group_type == 'image':
            return self._chunk_image_group(group, document_name, output_dir)
        elif group_type == 'table':
            return self._chunk_table_group(group, document_name, output_dir)
        elif group_type == 'equation':
            return self._chunk_equation_group(group, document_name, output_dir)
        else:
            logger.warning(f"未知内容类型: {group_type}")
            return []

    def _chunk_text_group(self, text_group: List[Dict], document_name: str) -> List[Chunk]:
        """对文本组进行切片

        Args:
            text_group: 文本内容组
            document_name: 文档名称

        Returns:
            文本切片列表
        """
        chunks = []

        if self.config.text_strategy == ChunkStrategy.DOCUMENT_STRUCTURE:
            # 基于文档结构的切片：考虑文本层级
            chunks = self._chunk_by_document_structure(text_group, document_name)
        elif self.config.text_strategy == ChunkStrategy.SEMANTIC:
            # 语义切片：基于段落和句子边界
            chunks = self._chunk_by_semantic_boundaries(text_group, document_name)
        elif self.config.text_strategy == ChunkStrategy.FIXED_SIZE:
            # 固定大小切片
            chunks = self._chunk_by_fixed_size(text_group, document_name)
        else:
            # 默认使用语义切片
            chunks = self._chunk_by_semantic_boundaries(text_group, document_name)

        return chunks

    def _chunk_by_document_structure(self, text_group: List[Dict], document_name: str) -> List[Chunk]:
        """基于文档结构进行切片

        考虑文本层级（标题、正文等），保持文档结构的完整性。
        """
        chunks = []
        current_chunk_content = []
        current_chunk_size = 0
        current_page = None

        for item in text_group:
            text = item.get('text', '').strip()
            if not text:
                continue

            text_level = item.get('text_level', 0)
            page_idx = item.get('page_idx')

            # 如果是标题（text_level > 0）且当前切片不为空，结束当前切片
            if (text_level > 0 and current_chunk_content and
                    current_chunk_size > self.config.min_chunk_size):
                chunk = self._create_text_chunk(
                    current_chunk_content, document_name, current_page
                )
                chunks.append(chunk)
                current_chunk_content = []
                current_chunk_size = 0

            # 添加到当前切片
            current_chunk_content.append(item)
            current_chunk_size += len(text)
            current_page = page_idx

            # 如果切片大小超过限制，结束当前切片
            if current_chunk_size >= self.config.max_chunk_size:
                chunk = self._create_text_chunk(
                    current_chunk_content, document_name, current_page
                )
                chunks.append(chunk)
                current_chunk_content = []
                current_chunk_size = 0

        # 处理最后一个切片
        if current_chunk_content:
            chunk = self._create_text_chunk(
                current_chunk_content, document_name, current_page
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_semantic_boundaries(self, text_group: List[Dict], document_name: str) -> List[Chunk]:
        """基于语义边界进行切片

        在句子和段落边界处切分，保持语义完整性。
        """
        chunks = []

        # 合并所有文本
        full_text = ""
        text_items = []
        for item in text_group:
            text = item.get('text', '').strip()
            if text:
                full_text += text + " "
                text_items.append(item)

        if not full_text.strip():
            return chunks

        # 按段落分割
        paragraphs = self.paragraph_pattern.split(full_text)

        current_chunk = ""
        current_items = []
        current_page = text_items[0].get('page_idx') if text_items else None

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # 如果添加当前段落会超过大小限制
            if (len(current_chunk) + len(paragraph) > self.config.max_chunk_size and
                    len(current_chunk) > self.config.min_chunk_size):

                # 创建当前切片
                if current_chunk.strip():
                    chunk = Chunk(
                        chunk_id=f"{document_name}_chunk_{self.chunk_counter}",
                        chunk_type=ChunkType.TEXT,
                        content=current_chunk.strip(),
                        page_idx=current_page,
                        chunk_idx=self.chunk_counter,
                        source_file=document_name,
                        parent_document=document_name,
                        metadata={
                            'chunk_strategy': 'semantic_boundaries',
                            'text_items_count': len(current_items),
                            'embedding_type': 'text',
                            'created_at': datetime.now().isoformat()
                        }
                    )
                    chunks.append(chunk)
                    self.chunk_counter += 1

                # 开始新切片
                current_chunk = paragraph
                current_items = []
            else:
                # 添加到当前切片
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # 处理最后一个切片
        if current_chunk.strip():
            chunk = Chunk(
                chunk_id=f"{document_name}_chunk_{self.chunk_counter}",
                chunk_type=ChunkType.TEXT,
                content=current_chunk.strip(),
                page_idx=current_page,
                chunk_idx=self.chunk_counter,
                source_file=document_name,
                parent_document=document_name,
                metadata={
                    'chunk_strategy': 'semantic_boundaries',
                    'text_items_count': len(current_items),
                    'embedding_type': 'text',
                    'created_at': datetime.now().isoformat()
                }
            )
            chunks.append(chunk)
            self.chunk_counter += 1

        return chunks

    def _chunk_by_fixed_size(self, text_group: List[Dict], document_name: str) -> List[Chunk]:
        """基于固定大小进行切片

        按照固定字符数进行切分，支持重叠。
        """
        chunks = []

        # 合并所有文本
        full_text = ""
        for item in text_group:
            text = item.get('text', '').strip()
            if text:
                full_text += text + " "

        if not full_text.strip():
            return chunks

        # 固定大小切片
        start = 0
        while start < len(full_text):
            end = start + self.config.max_chunk_size
            chunk_text = full_text[start:end]

            # 如果不是最后一个切片，尝试在句子边界处结束
            if end < len(full_text) and self.config.preserve_sentences:
                # 寻找最后一个句子结束符
                last_sentence_end = -1
                for match in self.sentence_pattern.finditer(chunk_text):
                    last_sentence_end = match.end()

                if last_sentence_end > self.config.min_chunk_size:
                    chunk_text = chunk_text[:last_sentence_end]
                    end = start + last_sentence_end

            if chunk_text.strip():
                chunk = Chunk(
                    chunk_id=f"{document_name}_chunk_{self.chunk_counter}",
                    chunk_type=ChunkType.TEXT,
                    content=chunk_text.strip(),
                    chunk_idx=self.chunk_counter,
                    start_pos=start,
                    end_pos=end,
                    source_file=document_name,
                    parent_document=document_name,
                    metadata={
                        'chunk_strategy': 'fixed_size',
                        'embedding_type': 'text',
                        'created_at': datetime.now().isoformat()
                    }
                )
                chunks.append(chunk)
                self.chunk_counter += 1

            # 移动到下一个位置，考虑重叠
            start = end - self.config.overlap_size
            if start <= 0:
                start = end

        return chunks

    def _create_text_chunk(self, text_items: List[Dict], document_name: str, page_idx: Optional[int]) -> Chunk:
        """创建文本切片"""
        content = "\n".join([item.get('text', '').strip() for item in text_items if item.get('text', '').strip()])

        # 提取元数据
        metadata = {
            'chunk_strategy': 'document_structure',
            'text_items_count': len(text_items),
            'embedding_type': 'text',
            'has_title': any(item.get('text_level', 0) > 0 for item in text_items),
            'created_at': datetime.now().isoformat()
        }

        # 添加文本层级信息
        text_levels = [item.get('text_level', 0) for item in text_items]
        if text_levels:
            metadata['text_levels'] = text_levels
            metadata['max_text_level'] = max(text_levels)

        chunk = Chunk(
            chunk_id=f"{document_name}_chunk_{self.chunk_counter}",
            chunk_type=ChunkType.TEXT,
            content=content,
            page_idx=page_idx,
            chunk_idx=self.chunk_counter,
            source_file=document_name,
            parent_document=document_name,
            metadata=metadata
        )

        self.chunk_counter += 1
        return chunk

    def _chunk_image_group(self, image_group: List[Dict], document_name: str, output_dir: str) -> List[Chunk]:
        """对图片组进行切片

        图片通常作为单独的切片处理，包含图片路径和相关描述信息。
        对于图片和表格类切片，需要执行双入库操作：
        1. 以图像作为embedding输入，metadata保存图像描述和Base64参数
        2. 以image_caption作为embedding输入，metadata保存图像描述和Base64参数
        """
        chunks = []

        for item in image_group:
            img_path = item.get('img_path')
            if not img_path:
                continue

            # 构建完整的图片路径
            full_img_path = os.path.join(output_dir, img_path)

            # 获取图片标题，确保是字符串类型
            caption_raw = item.get('image_caption', '')
            if isinstance(caption_raw, list):
                caption = '; '.join(caption_raw) if caption_raw else ''
            else:
                caption = caption_raw or ''

            # 构建内容描述
            content_parts = []

            # 添加图片标题
            if self.config.include_image_caption and caption:
                content_parts.append(f"图片标题: {caption}")

            # 添加图片描述
            content_parts.append(f"图片路径: {img_path}")

            # 注意：当前 content_list.json 中的图片条目不包含 text 字段
            # OCR 文本实际存储在 model.json 文件中，但当前实现未读取该文件
            # 如果需要 OCR 文本，需要额外读取 model.json 文件

            content = "\n".join(content_parts)

            # 第一个切片：以图像作为embedding输入
            image_chunk = Chunk(
                chunk_id=f"{document_name}_img_{self.chunk_counter}",
                chunk_type=ChunkType.IMAGE,
                content=content,
                content_path=full_img_path,
                page_idx=item.get('page_idx'),
                chunk_idx=self.chunk_counter,
                source_file=document_name,
                parent_document=document_name,
                metadata={
                    'chunk_strategy': 'image_visual',
                    'image_path': img_path,
                    'has_caption': bool(caption),
                    'has_ocr_text': False,
                    'embedding_type': 'visual',  # 标记为视觉embedding
                    'image_caption': caption,
                    'created_at': datetime.now().isoformat()
                }
            )

            chunks.append(image_chunk)
            self.chunk_counter += 1

            # 第二个切片：以image_caption作为embedding输入（如果有标题）
            if caption:
                caption_chunk = Chunk(
                    chunk_id=f"{document_name}_img_caption_{self.chunk_counter}",
                    chunk_type=ChunkType.IMAGE,
                    content=caption,  # 直接使用标题作为内容
                    content_path=full_img_path,
                    page_idx=item.get('page_idx'),
                    chunk_idx=self.chunk_counter,
                    source_file=document_name,
                    parent_document=document_name,
                    metadata={
                        'chunk_strategy': 'image_caption',
                        'image_path': img_path,
                        'has_caption': True,
                        'has_ocr_text': False,
                        'embedding_type': 'text',  # 标记为文本embedding
                        'image_caption': caption,
                        'related_visual_chunk': f"{document_name}_img_{self.chunk_counter - 1}",  # 关联的视觉切片ID
                        'created_at': datetime.now().isoformat()
                    }
                )

                chunks.append(caption_chunk)
                self.chunk_counter += 1

        return chunks

    def _chunk_table_group(self, table_group: List[Dict], document_name: str, output_dir: str) -> List[Chunk]:
        """对表格组进行切片

        表格通常作为单独的切片处理，包含表格的HTML内容和相关描述信息。
        对于表格类切片，需要执行双入库操作：
        1. 以表格图像作为embedding输入，metadata保存表格描述和Base64参数
        2. 以表格标题和内容作为embedding输入，metadata保存表格描述和Base64参数
        """
        chunks = []

        for item in table_group:
            table_body = item.get('table_body')
            if not table_body:
                continue

            # 获取表格相关信息
            caption = item.get('table_caption', [])
            footnote = item.get('table_footnote', [])
            img_path = item.get('img_path')

            # 构建内容描述
            content_parts = []

            # 添加表格标题
            if self.config.include_table_caption and caption:
                content_parts.append(f"表格标题: {'; '.join(caption)}")

            # 添加表格内容
            content_parts.append("表格内容:")
            content_parts.append(table_body)

            # 添加表格脚注
            if footnote:
                content_parts.append(f"表格脚注: {'; '.join(footnote)}")

            content = "\n".join(content_parts)

            # 构建表格图片路径（如果存在）
            full_img_path = None
            if img_path:
                full_img_path = os.path.join(output_dir, img_path)

            # 第一个切片：以表格图像作为embedding输入（如果有图片）
            if img_path:
                table_visual_chunk = Chunk(
                    chunk_id=f"{document_name}_table_{self.chunk_counter}",
                    chunk_type=ChunkType.TABLE,
                    content=content,
                    content_path=full_img_path,
                    page_idx=item.get('page_idx'),
                    chunk_idx=self.chunk_counter,
                    source_file=document_name,
                    parent_document=document_name,
                    metadata={
                        'chunk_strategy': 'table_visual',
                        'table_image_path': img_path,
                        'has_caption': bool(caption),
                        'has_footnote': bool(footnote),
                        'table_html_length': len(table_body),
                        'embedding_type': 'visual',  # 标记为视觉embedding
                        'table_caption': '; '.join(caption) if caption else '',
                        'table_content': table_body,
                        'created_at': datetime.now().isoformat()
                    }
                )

                chunks.append(table_visual_chunk)
                self.chunk_counter += 1

            # 第二个切片：以表格文本内容作为embedding输入
            table_text_chunk = Chunk(
                chunk_id=f"{document_name}_table_text_{self.chunk_counter}",
                chunk_type=ChunkType.TABLE,
                content=content,
                content_path=full_img_path,
                page_idx=item.get('page_idx'),
                chunk_idx=self.chunk_counter,
                source_file=document_name,
                parent_document=document_name,
                metadata={
                    'chunk_strategy': 'table_text',
                    'table_image_path': img_path,
                    'has_caption': bool(caption),
                    'has_footnote': bool(footnote),
                    'table_html_length': len(table_body),
                    'embedding_type': 'text',  # 标记为文本embedding
                    'table_caption': '; '.join(caption) if caption else '',
                    'table_content': table_body,
                    'related_visual_chunk': f"{document_name}_table_{self.chunk_counter - 1}" if img_path else None,
                    # 关联的视觉切片ID
                    'created_at': datetime.now().isoformat()
                }
            )

            chunks.append(table_text_chunk)
            self.chunk_counter += 1

        return chunks

    def _chunk_equation_group(self, equation_group: List[Dict], document_name: str, output_dir: str) -> List[Chunk]:
        """对数学公式组进行切片

        数学公式通常作为单独的切片处理，包含公式的LaTeX内容和相关信息。
        """
        chunks = []

        for item in equation_group:
            latex_text = item.get('text', '')
            if not latex_text:
                continue

            # 构建内容描述
            content_parts = []

            # 添加公式类型说明
            content_parts.append("数学公式:")
            content_parts.append(latex_text)

            content = "\n".join(content_parts)

            # 构建公式图片路径（如果存在）
            img_path = item.get('img_path')
            full_img_path = None
            if img_path:
                full_img_path = os.path.join(output_dir, img_path)

            chunk = Chunk(
                chunk_id=f"{document_name}_equation_{self.chunk_counter}",
                chunk_type=ChunkType.EQUATION,
                content=content,
                content_path=full_img_path,
                page_idx=item.get('page_idx'),
                chunk_idx=self.chunk_counter,
                source_file=document_name,
                parent_document=document_name,
                metadata={
                    'chunk_strategy': 'equation_single',
                    'equation_image_path': img_path,
                    'text_format': item.get('text_format', 'latex'),
                    'has_latex_text': bool(latex_text),
                    'has_equation_image': bool(img_path),
                    'latex_length': len(latex_text),
                    'created_at': datetime.now().isoformat()
                }
            )

            chunks.append(chunk)
            self.chunk_counter += 1

        return chunks

    def _post_process_chunks(self):
        """后处理切片

        添加切片间的关联信息，优化切片质量。
        """
        if not self.chunks:
            return

        # 添加相邻切片的关联信息
        for i, chunk in enumerate(self.chunks):
            related_chunks = []

            # 添加前一个切片
            if i > 0:
                related_chunks.append(self.chunks[i - 1].chunk_id)

            # 添加后一个切片
            if i < len(self.chunks) - 1:
                related_chunks.append(self.chunks[i + 1].chunk_id)

            chunk.related_chunks = related_chunks

            # 添加全局元数据
            chunk.metadata.update({
                'total_chunks': len(self.chunks),
                'chunk_position': i + 1
            })