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
from pathlib import Path
from ..chunking.chunk_agent import Chunk, ChunkType

logger = logging.getLogger(__name__)


class ChromaAgent:
    """ChromaDB代理类
    
    负责处理chunks与ChromaDB的交互，包括:
    - 元数据增强
    - Base64图像编码
    - 重复检查
    - 数据库操作
    - 切片去重处理
    - 文档详情查询
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
    
    def deduplicate_chunks_for_display(self, chunks: List[Dict]) -> List[Dict]:
        """对知识库管理页面的切片进行去重处理，合并双入库的切片
        
        Args:
            chunks: 原始切片列表
            
        Returns:
            去重后的切片信息列表，每个元素包含:
            - chunk_type: 切片类型
            - page_idx: 页面索引
            - chunk_idx_display: 显示用的切片索引（如"5-6"表示双入库）
            - primary_chunk: 主切片（用于显示）
            - secondary_chunk: 次切片（如果存在）
            - processed_content: 处理后的内容信息（包含多模态数据）
        """
        if not chunks:
            return []
        
        deduplicated = []
        processed_chunks = set()
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', '') or chunk.get('id', '')
            if chunk_id in processed_chunks:
                continue
                
            chunk_type = chunk['chunk_type']
            metadata = chunk.get('metadata', {})
            
            # 检查是否是双入库的切片（仅对image和table类型进行双入库合并）
            if chunk_type in ['image', 'table']:
                embedding_type = metadata.get('embedding_type', '')
                
                if embedding_type == 'visual':
                    # 这是视觉embedding切片，寻找对应的文本embedding切片
                    text_chunk = None
                    
                    # 使用更稳定的标识符来查找配对切片
                    source_file = metadata.get('source_file', '') or metadata.get('parent_document', '')
                    page_idx = chunk['page_number']
                    
                    # 构建基础标识符
                    if chunk_type == 'image':
                        image_path = metadata.get('image_path', '')
                        base_identifier = f"{chunk_type}_{source_file}_{page_idx}_{image_path}"
                    else:  # table
                        table_image_path = metadata.get('table_image_path', '')
                        base_identifier = f"{chunk_type}_{source_file}_{page_idx}_{table_image_path}"
                    
                    # 查找对应的文本embedding切片
                    for other_chunk in chunks:
                        other_metadata = other_chunk.get('metadata', {})
                        if (other_chunk['chunk_type'] == chunk_type and
                            other_chunk['page_number'] == page_idx and
                            other_metadata.get('embedding_type') == 'text' and
                            (other_metadata.get('source_file') == source_file or 
                             other_metadata.get('parent_document') == source_file)):
                            
                            # 检查是否是同一个图像/表格的文本版本
                            if chunk_type == 'image':
                                other_image_path = other_metadata.get('image_path', '')
                                if other_image_path == image_path:
                                    text_chunk = other_chunk
                                    break
                            else:  # table
                                other_table_image_path = other_metadata.get('table_image_path', '')
                                if other_table_image_path == table_image_path:
                                    text_chunk = other_chunk
                                    break
                    
                    if text_chunk:
                        # 找到了配对的切片，合并显示
                        processed_content = self._process_multimodal_content(chunk, chunk_type)
                        chunk_info = {
                            'chunk_type': chunk_type,
                            'page_idx': chunk['page_number'],
                            'chunk_idx_display': f"{chunk['chunk_index']}-{text_chunk['chunk_index']}",
                            'primary_chunk': chunk,  # 使用视觉切片作为主切片
                            'secondary_chunk': text_chunk,
                            'processed_content': processed_content
                        }
                        processed_chunks.add(chunk_id)
                        processed_chunks.add(text_chunk.get('chunk_id', '') or text_chunk.get('id', ''))
                    else:
                        # 没有找到配对切片，单独显示
                        processed_content = self._process_multimodal_content(chunk, chunk_type)
                        chunk_info = {
                            'chunk_type': chunk_type,
                            'page_idx': chunk['page_number'],
                            'chunk_idx_display': str(chunk['chunk_index']),
                            'primary_chunk': chunk,
                            'processed_content': processed_content
                        }
                        processed_chunks.add(chunk_id)
                        
                elif embedding_type == 'text':
                    # 这是文本embedding切片，检查是否已经被处理过
                    if chunk_id not in processed_chunks:
                        # 单独的文本切片（没有对应的视觉切片）
                        processed_content = self._process_multimodal_content(chunk, chunk_type)
                        chunk_info = {
                            'chunk_type': chunk_type,
                            'page_idx': chunk['page_number'],
                            'chunk_idx_display': str(chunk['chunk_index']),
                            'primary_chunk': chunk,
                            'processed_content': processed_content
                        }
                        processed_chunks.add(chunk_id)
                    else:
                        continue
                else:
                    # 没有embedding_type标记的切片，单独显示
                    processed_content = self._process_multimodal_content(chunk, chunk_type)
                    chunk_info = {
                        'chunk_type': chunk_type,
                        'page_idx': chunk['page_number'],
                        'chunk_idx_display': str(chunk['chunk_index']),
                        'primary_chunk': chunk,
                        'processed_content': processed_content
                    }
                    processed_chunks.add(chunk_id)
            else:
                # 非image/table类型的切片（如text类型），直接显示，不进行双入库合并
                processed_content = self._process_multimodal_content(chunk, chunk_type)
                chunk_info = {
                    'chunk_type': chunk_type,
                    'page_idx': chunk['page_number'],
                    'chunk_idx_display': str(chunk['chunk_index']),
                    'primary_chunk': chunk,
                    'processed_content': processed_content
                }
                processed_chunks.add(chunk_id)
            
            deduplicated.append(chunk_info)
        
        return deduplicated
    
    def get_document_detail(
        self,
        file_id: str,
        include_embeddings: bool = False,
        chunk_limit: Optional[int] = None,
        deduplicate: bool = True
    ) -> Dict[str, Any]:
        """获取文档的详细切片信息
        
        Args:
            file_id: 文件ID或文件名
            include_embeddings: 是否包含嵌入向量数据
            chunk_limit: 限制返回的切片数量，None表示返回所有
            deduplicate: 是否对切片进行去重处理（合并双入库的图像和表格切片）
            
        Returns:
            包含文档详细信息和切片数据的字典
        """
        try:
            # 通过ChromaDB provider获取所有文档
            if not (self.vector_store and 
                    hasattr(self.vector_store, '_provider') and 
                    hasattr(self.vector_store._provider, '_collection')):
                return {
                    "error": "Vector store未初始化或不支持直接文档访问",
                    "file_info": None,
                    "chunks": [],
                    "total_chunks": 0,
                    "returned_chunks": 0
                }
            
            collection = self.vector_store._provider._collection
            
            # 获取所有文档的元数据，用于查找匹配的文档
            all_docs = collection.get(include=['metadatas'])
            
            if not all_docs['metadatas']:
                return {
                    "error": "知识库为空",
                    "file_info": None,
                    "chunks": [],
                    "total_chunks": 0,
                    "returned_chunks": 0
                }
            
            # 查找匹配的文档ID
            matching_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                # 检查多个可能的字段来匹配文件
                doc_source = (
                    metadata.get('parent_document') or 
                    metadata.get('source_file') or 
                    metadata.get('source') or 
                    metadata.get('file_path', '')
                )
                
                # 支持多种匹配方式
                if (doc_source == file_id or  # 完全匹配
                    doc_source.endswith(file_id) or  # 文件名匹配
                    Path(doc_source).name == file_id or  # 提取文件名匹配
                    Path(doc_source).stem == Path(file_id).stem):  # stem匹配
                    matching_ids.append(all_docs['ids'][i])
            
            if not matching_ids:
                return {
                    "error": f"未找到文档: {file_id}",
                    "file_info": None,
                    "chunks": [],
                    "total_chunks": 0,
                    "returned_chunks": 0
                }
            
            # 获取详细的文档数据
            include_fields = ['documents', 'metadatas']
            if include_embeddings:
                include_fields.append('embeddings')
            
            detailed_docs = collection.get(
                ids=matching_ids,
                include=include_fields
            )
            
            # 对切片进行排序和处理
            chunks_data = []
            file_metadata = {}
            
            # 构建切片数据列表
            for i, doc_id in enumerate(detailed_docs['ids']):
                metadata = detailed_docs['metadatas'][i] if detailed_docs['metadatas'] else {}
                content = detailed_docs['documents'][i] if detailed_docs['documents'] else ""
                
                # 如果是第一个切片，保存文件级别的元数据
                if i == 0:
                    file_metadata = metadata.copy()
                
                chunk_data = {
                    'chunk_id': metadata.get('chunk_id', doc_id),
                    'chunk_type': metadata.get('chunk_type', 'text'),
                    'page_idx': metadata.get('page_idx') or metadata.get('page_number', 0),
                    'chunk_idx': metadata.get('chunk_idx', i),
                    'content': content,
                    'metadata': metadata,
                    'doc_id': doc_id  # 保留原始文档ID
                }
                
                # 添加嵌入向量（如果需要）
                if include_embeddings and detailed_docs.get('embeddings'):
                    chunk_data['embedding'] = detailed_docs['embeddings'][i]
                else:
                    chunk_data['embedding'] = None
                
                chunks_data.append(chunk_data)
            
            # 按页面和切片索引排序（参考Vision_RAG的排序逻辑）
            chunks_data.sort(key=lambda x: (x['page_idx'], x['chunk_idx']))
            
            # 应用切片限制
            total_chunks = len(chunks_data)
            if chunk_limit is not None:
                chunks_data = chunks_data[:chunk_limit]
            
            # 统计切片类型
            chunk_types = {}
            for chunk in chunks_data:
                chunk_type = chunk['chunk_type']
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # 获取页面范围
            pages = set(chunk['page_idx'] for chunk in chunks_data if chunk['page_idx'] is not None)
            page_range = None
            if pages:
                if len(pages) == 1:
                    page_range = f"第 {list(pages)[0]} 页"
                else:
                    page_range = f"第 {min(pages)} - {max(pages)} 页"
            
            # 构建文件信息
            file_source = (
                file_metadata.get('parent_document') or 
                file_metadata.get('source_file') or 
                file_metadata.get('source') or 
                file_id
            )
            
            file_info = {
                "file_id": file_id,
                "file_name": Path(file_source).name if file_source else file_id,
                "total_chunks": total_chunks,
                "chunk_types": chunk_types,
                "page_range": page_range,
                "file_type": Path(file_source).suffix.lstrip('.') if '.' in file_source else 'unknown',
                "created_at": file_metadata.get('created_at', file_metadata.get('timestamp')),
                "metadata": file_metadata
            }
            
            # 转换为标准格式的切片数据
            formatted_chunks = []
            for chunk in chunks_data:
                formatted_chunk = {
                    "chunk_id": chunk['doc_id'],
                    "chunk_index": chunk['chunk_idx'],
                    "content": chunk['content'],
                    "chunk_type": chunk['chunk_type'],
                    "page_number": chunk['page_idx'],
                    "similarity_score": None,  # 仅在搜索时有值
                    "metadata": chunk['metadata'],
                    "embedding": chunk['embedding']
                }
                formatted_chunks.append(formatted_chunk)
            
            # 在返回结果前应用去重处理
            if deduplicate:
                deduplicated_chunks = self.deduplicate_chunks_for_display(formatted_chunks)
                
                # 重新计算统计信息
                chunk_types_dedup = {}
                for chunk_info in deduplicated_chunks:
                    chunk_type = chunk_info['chunk_type']
                    chunk_types_dedup[chunk_type] = chunk_types_dedup.get(chunk_type, 0) + 1
                
                # 更新文件信息
                file_info["chunk_types"] = chunk_types_dedup
                file_info["total_chunks_after_dedup"] = len(deduplicated_chunks)
                file_info["original_total_chunks"] = total_chunks
                
                return {
                    "file_info": file_info,
                    "chunks": deduplicated_chunks,
                    "total_chunks": len(deduplicated_chunks),
                    "returned_chunks": len(deduplicated_chunks),
                    "deduplicated": True
                }
            else:
                return {
                    "file_info": file_info,
                    "chunks": formatted_chunks,
                    "total_chunks": total_chunks,
                    "returned_chunks": len(formatted_chunks),
                    "deduplicated": False
                }
                
        except Exception as e:
            logger.error(f"Failed to retrieve document details for {file_id}: {e}")
            return {
                "error": f"获取文档详情失败: {str(e)}",
                "file_info": None,
                "chunks": [],
                "total_chunks": 0,
                "returned_chunks": 0
            }

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
            # content = chunk_data.get('content', '')
            # metadata = chunk_data.get('metadata', {})
            # doc_id = self._generate_content_id(content, metadata)
            # chunk_data['doc_id'] = doc_id
            if 'chunk_id' in chunk_data['metadata']:
                chunk_data['doc_id'] = chunk_data['metadata']['chunk_id']
            else:
                # fallback：生成UUID
                import uuid
                chunk_data['doc_id'] = str(uuid.uuid4())
    
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

    def _process_multimodal_content(self, chunk: Dict, chunk_type: str) -> Dict[str, Any]:
        """处理多模态内容，提取和验证图片、表格等数据

        Args:
            chunk: 切片数据
            chunk_type: 切片类型

        Returns:
            处理后的内容信息字典
        """
        processed_content = {
            'content_type': chunk_type,
            'text_content': chunk.get('content', ''),
            'has_image': False,
            'image_info': {},
            'error_messages': []
        }

        metadata = chunk.get('metadata', {})

        if chunk_type == 'image':
            # 处理图片内容
            image_info = self._process_image_content(chunk, metadata)
            processed_content.update(image_info)

        elif chunk_type == 'table':
            # 处理表格内容
            table_info = self._process_table_content(chunk, metadata)
            processed_content.update(table_info)

        else:
            # 处理文本内容
            content = chunk.get('content', '')
            if content:
                # 限制文本长度用于显示
                display_content = content[:500] + "..." if len(content) > 500 else content
                processed_content['display_content'] = display_content
                processed_content['full_content'] = content

        return processed_content

    def _process_image_content(self, chunk: Dict, metadata: Dict) -> Dict[str, Any]:
        """处理图片内容

        Args:
            chunk: 切片数据
            metadata: 元数据

        Returns:
            图片处理结果
        """
        result = {
            'has_image': False,
            'image_info': {},
            'error_messages': []
        }

        try:
            # 首先检查是否有original_content (base64数据)
            if 'original_content' in metadata:
                try:
                    original_content = metadata['original_content']
                    if original_content:
                        # 验证Base64数据
                        image_data = base64.b64decode(original_content)
                        result['has_image'] = True
                        result['image_info'] = {
                            'source': 'base64',
                            'data_size': len(image_data),
                            'base64_data': original_content
                        }
                        return result
                except Exception as b64_error:
                    result['error_messages'].append(f"解码base64图片失败: {b64_error}")

            # 如果base64处理失败，尝试从本地路径加载
            if 'image_path' in metadata:
                image_path = metadata['image_path']
                # 如果是相对路径，尝试构建绝对路径
                if not os.path.isabs(image_path) and chunk.get('content_path'):
                    # 使用content_path作为绝对路径
                    full_image_path = chunk['content_path']
                else:
                    full_image_path = image_path

                if os.path.exists(full_image_path):
                    result['has_image'] = True
                    result['image_info'] = {
                        'source': 'file_path',
                        'path': full_image_path,
                        'original_path': image_path,
                        'file_size': os.path.getsize(full_image_path)
                    }
                else:
                    result['error_messages'].append(f"图片文件不存在: {full_image_path}")

            # 如果都失败了，记录错误
            if not result['has_image']:
                result['error_messages'].append("无法获取图片数据：缺少有效的图片源")

        except Exception as img_error:
            result['error_messages'].append(f"处理图片时出错: {img_error}")

        return result

    def _process_table_content(self, chunk: Dict, metadata: Dict) -> Dict[str, Any]:
        """处理表格内容

        Args:
            chunk: 切片数据
            metadata: 元数据

        Returns:
            表格处理结果
        """
        result = {
            'has_image': False,
            'table_info': {},
            'error_messages': []
        }

        try:
            # 如果有表格图片，处理图片
            if 'table_image_path' in metadata or 'image_path' in metadata:
                # 优先使用table_image_path，如果没有则使用image_path
                image_path = metadata.get('table_image_path') or metadata.get('image_path')
                # 如果是相对路径，尝试构建绝对路径
                if not os.path.isabs(image_path) and chunk.get('content_path'):
                    # 使用content_path作为绝对路径
                    full_image_path = chunk['content_path']
                else:
                    full_image_path = image_path

                if os.path.exists(full_image_path):
                    result['has_image'] = True
                    result['table_info'] = {
                        'image_source': 'file_path',
                        'image_path': full_image_path,
                        'original_path': image_path,
                        'file_size': os.path.getsize(full_image_path)
                    }
                else:
                    result['error_messages'].append(f"表格图片文件不存在: {full_image_path}")

            # 处理表格文本内容
            content = chunk.get('content', '')
            if content and content.strip():
                result['table_info']['text_content'] = content
                result['table_info']['has_text'] = True
            else:
                result['table_info']['has_text'] = False

        except Exception as table_error:
            result['error_messages'].append(f"处理表格时出错: {table_error}")

        return result
