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
    
    def __init__(self, vector_store=None, config: Optional[Dict[str, Any]] = None):
        """初始化ChromaDB代理
        
        Args:
            vector_store: VectorStore实例
            config: 配置参数字典
        """
        self.vector_store = vector_store
        self.config = config or {}
        logger.info("ChromaAgent initialized")
    
    def _find_image_path(self, output_dir: str, source_file: str, image_path: str) -> Optional[str]:
        """查找图片文件路径，参考engine.py的逻辑"""
        from pathlib import Path
        
        output_path = Path(output_dir)
        
        # 方法1：尝试基本路径
        basic_path = output_path / image_path
        if basic_path.exists():
            return str(basic_path)
        
        # 方法2：尝试子目录路径（MinerU 2.0结构）
        subdir = output_path / source_file
        if subdir.exists():
            for method in ['auto', 'ocr']:  # 尝试不同的方法目录
                method_path = subdir / method / image_path
                if method_path.exists():
                    return str(method_path)
        
        # 方法3：尝试传统的嵌套结构
        for method in ['auto', 'ocr']:
            nested_path = output_path / source_file / source_file / method / image_path
            if nested_path.exists():
                return str(nested_path)
        
        return None
    
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
                'content': chunk['content'],
                'metadata': {
                    'source_file': source_file_name,
                    'chunk_type': chunk['chunk_type'].value,
                    'chunk_id': chunk['chunk_id'],
                    'page_idx': chunk['page_idx'],
                    'chunk_idx': chunk['chunk_idx'],
                    'parent_document': chunk['parent_document'],
                    **chunk['metadata']
                }
            }
    
            # 生成doc_id
            content = chunk_data.get('content', '')
            metadata = chunk_data.get('metadata', {})
            doc_id = self._generate_content_id(content, metadata)
            chunk_data['doc_id'] = doc_id
    
            # 直接在这里处理图像和表格的 Base64 数据
            if chunk['chunk_type'].value in ['image', 'table'] and chunk['content_path']:
                try:
                    with open(chunk['content_path'], 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                        chunk_data['original_content'] = image_data
                except Exception as e:
                    logger.error(f"读取图片文件失败 {chunk.content_path}: {e}")
                    # 使用新的路径查找方法
                    image_path = chunk['metadata'].get('image_path') or chunk['metadata'].get('table_image_path')
                    if image_path:
                        source_file = chunk['source_file'] or source_file_name.split('.')[0]
                        output_dir = self.config.get('output_dir', 'output_dir')
                        
                        # 使用新的路径查找方法
                        full_path = self._find_image_path(output_dir, source_file, image_path)
                        if full_path:
                            try:
                                with open(full_path, 'rb') as f:
                                    image_data = base64.b64encode(f.read()).decode('utf-8')
                                    chunk_data['original_content'] = image_data
                            except Exception as e2:
                                logger.error(f"使用查找到的路径读取图片文件也失败: {e2}")
    
            # 检查重复和分类
            if not self.vector_store or not self.vector_store._provider or not self.vector_store._provider._collection:
                # 如果没有vector_store，默认为新项
                results['new_items'].append(chunk_data)
            else:
                collection = self.vector_store._provider._collection
                # 如果是重新解析模式，所有切片都视为需要覆盖的重复项
                if is_reparse:
                    # 为重新解析的切片添加existing_doc字段
                    try:
                        existing = collection.get(
                            where={
                                "$and": [
                                    {"source_file": source_file_name},
                                    {"chunk_id": chunk['chunk_id']}
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
                        existing = collection.get(
                            where={
                                "$and": [
                                    {"source_file": source_file_name},
                                    {"chunk_id": chunk['chunk_id']}
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

    def _generate_content_id(self, content: str, metadata: Dict) -> str:
        """生成基于内容的唯一ID"""
        import hashlib
        
        # 构建用于生成ID的字符串
        id_components = [
            content,
            metadata.get('source_file', ''),
            str(metadata.get('page_idx', '')),
            metadata.get('chunk_type', '')
        ]
        
        id_string = '|'.join(str(comp) for comp in id_components)
        return hashlib.md5(id_string.encode('utf-8')).hexdigest()    
    
    def add_chunks_to_db(
        self,
        chunks_data: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> bool:
        """将chunks添加到数据库"""
        if not self.vector_store or not self.vector_store._provider or not self.vector_store._provider._collection:
            logger.error("VectorStore未初始化")
            return False
            
        try:
            collection = self.vector_store._provider._collection
            for chunk_data in chunks_data:
                collection.add(
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
        """更新数据库中的chunks"""
        if not self.vector_store or not self.vector_store._provider or not self.vector_store._provider._collection:
            logger.error("VectorStore未初始化")
            return False
            
        try:
            collection = self.vector_store._provider._collection
            for chunk_data in chunks_data:
                if 'existing_doc' in chunk_data:
                    # 更新现有文档
                    collection.update(
                        ids=[chunk_data['existing_doc']['id']],
                        documents=[chunk_data['content']],
                        metadatas=[chunk_data['metadata']]
                    )
            logger.info(f"成功更新 {len(chunks_data)} 个chunks")
            return True
        except Exception as e:
            logger.error(f"更新chunks失败: {e}")
            return False
