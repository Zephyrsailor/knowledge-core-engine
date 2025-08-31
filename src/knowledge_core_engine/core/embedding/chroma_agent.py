#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB代理模块

该模块负责处理chunks的向量化存储、重复检查和数据库操作。
专门处理与ChromaDB相关的业务逻辑，包括Base64编码、元数据增强等。

Author: fanjs
Date: 2025-08-28
"""

import os
import base64
import logging
from typing import Dict, List, Any, Optional
from ..chunking.chunk_agent import Chunk, ChunkType

logger = logging.getLogger(__name__)


class ChromaAgent:
    """ChromaDB代理类
    
    负责处理chunks与ChromaDB的交互，包括:
    - 元数据增强
    - Base64图像编码
    - 重复检查
    - 数据库操作
    """
    
    def __init__(self, db_manager=None, config: Optional[Dict[str, Any]] = None):
        """初始化ChromaDB代理
        
        Args:
            db_manager: ChromaDB管理器实例
            config: 配置参数字典
        """
        self.db_manager = db_manager
        self.config = config or {}
        logger.info("ChromaAgent initialized")
    
    def process_chunks_for_service(
        self,
        chunks: List[Chunk],
        source_file_name: str,
        is_reparse: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """处理 chunks 的元数据增强、Base64 编码和数据库重复检查。"""
        results = {'duplicates': [], 'new_items': []}
    
        for chunk in chunks:
            chunk_data = {
                'content': chunk.content,
                'metadata': {
                    'source_file': source_file_name,
                    'chunk_type': chunk.chunk_type.value,
                    'chunk_id': chunk.chunk_id,
                    'page_idx': chunk.page_idx,
                    'chunk_idx': chunk.chunk_idx,
                    'parent_document': chunk.parent_document,
                    **chunk.metadata
                }
            }
    
            # 直接在这里处理图像和表格的 Base64 数据
            if chunk.chunk_type.value in ['image', 'table'] and chunk.content_path:
                try:
                    with open(chunk.content_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                        chunk_data['original_content'] = image_data
                except Exception as e:
                    logger.error(f"读取图片文件失败 {chunk.content_path}: {e}")
                    # 尝试相对路径和不同方法
                    image_path = chunk.metadata.get('image_path') or chunk.metadata.get('table_image_path')
                    if image_path:
                        source_file = chunk.source_file or source_file_name.split('.')[0]
                        for method in ['auto', 'ocr']:
                            base_dir = os.path.join(
                                self.config.get('output_dir', 'output_dir'), 
                                self.config.get('cls_dir', 'cls_dir'), 
                                source_file, 
                                source_file, 
                                method
                            )
                            full_path = os.path.join(base_dir, image_path)
                            if os.path.exists(full_path):
                                try:
                                    with open(full_path, 'rb') as f:
                                        image_data = base64.b64encode(f.read()).decode('utf-8')
                                        chunk_data['original_content'] = image_data
                                    break
                                except Exception as e2:
                                    logger.error(f"使用相对路径读取图片文件也失败: {e2}")
    
            # 检查重复和分类
            if not self.db_manager:
                # 如果没有数据库管理器，默认为新项
                results['new_items'].append(chunk_data)
            else:
                # 如果是重新解析模式，所有切片都视为需要覆盖的重复项
                if is_reparse:
                    # 为重新解析的切片添加existing_doc字段
                    try:
                        existing = self.db_manager.collection.get(
                            where={
                                "$and": [
                                    {"source_file": source_file_name},
                                    {"chunk_id": chunk.chunk_id}
                                ]
                            }
                        )
                        if existing and len(existing.get('ids', [])) > 0:
                            # 构造existing_doc结构
                            existing_doc = {
                                'id': existing['ids'][0],
                                'document': existing['documents'][0],
                                'metadata': existing['metadatas'][0]
                            }
                            chunk_data['existing_doc'] = existing_doc
                    except:
                        # 如果查询失败，创建一个默认的existing_doc
                        chunk_data['existing_doc'] = {
                            'id': 'unknown',
                            'document': '无法获取已存在文档信息',
                            'metadata': {}
                        }
                    results['duplicates'].append(chunk_data)
                else:
                    # 正常模式下检查数据库中是否存在
                    try:
                        existing = self.db_manager.collection.get(
                            where={
                                "$and": [
                                    {"source_file": source_file_name},
                                    {"chunk_id": chunk.chunk_id}
                                ]
                            }
                        )
                        if existing and len(existing.get('ids', [])) > 0:
                            # 构造existing_doc结构
                            existing_doc = {
                                'id': existing['ids'][0],
                                'document': existing['documents'][0],
                                'metadata': existing['metadatas'][0]
                            }
                            chunk_data['existing_doc'] = existing_doc
                            results['duplicates'].append(chunk_data)
                        else:
                            results['new_items'].append(chunk_data)
                    except Exception as e:
                        logger.error(f"数据库查询失败: {e}")
                        # 如果查询失败，默认为新项目
                        results['new_items'].append(chunk_data)
    
        return results
    
    def add_chunks_to_db(
        self,
        chunks_data: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> bool:
        """将chunks添加到数据库
        
        Args:
            chunks_data: 处理后的chunks数据列表
            collection_name: 集合名称
            
        Returns:
            是否成功添加
        """
        if not self.db_manager:
            logger.error("数据库管理器未初始化")
            return False
            
        try:
            # 这里可以根据实际的ChromaDB API进行实现
            # 示例实现，需要根据具体的db_manager接口调整
            for chunk_data in chunks_data:
                self.db_manager.add(
                    documents=[chunk_data['content']],
                    metadatas=[chunk_data['metadata']],
                    ids=[chunk_data['metadata']['chunk_id']]
                )
            logger.info(f"成功添加 {len(chunks_data)} 个chunks到数据库")
            return True
        except Exception as e:
            logger.error(f"添加chunks到数据库失败: {e}")
            return False
    
    def update_chunks_in_db(
        self,
        chunks_data: List[Dict[str, Any]]
    ) -> bool:
        """更新数据库中的chunks
        
        Args:
            chunks_data: 需要更新的chunks数据列表
            
        Returns:
            是否成功更新
        """
        if not self.db_manager:
            logger.error("数据库管理器未初始化")
            return False
            
        try:
            for chunk_data in chunks_data:
                if 'existing_doc' in chunk_data:
                    # 更新现有文档
                    self.db_manager.update(
                        ids=[chunk_data['existing_doc']['id']],
                        documents=[chunk_data['content']],
                        metadatas=[chunk_data['metadata']]
                    )
            logger.info(f"成功更新 {len(chunks_data)} 个chunks")
            return True
        except Exception as e:
            logger.error(f"更新chunks失败: {e}")
            return False