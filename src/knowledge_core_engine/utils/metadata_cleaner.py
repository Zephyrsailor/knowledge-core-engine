"""
元数据清理工具

确保元数据值符合向量数据库的要求
"""

import json
from typing import Dict, Any, Union


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理元数据，确保所有值都是简单类型
    
    ChromaDB 只支持 str, int, float, bool, None 类型的元数据值
    
    Args:
        metadata: 原始元数据字典
        
    Returns:
        清理后的元数据字典
    """
    if not metadata:
        return {}
    
    cleaned = {}
    
    for key, value in metadata.items():
        # 跳过None值 - ChromaDB不支持None
        if value is None:
            continue
            
        # 简单类型直接保留
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
            
        # 列表转换为逗号分隔的字符串
        elif isinstance(value, list):
            if not value:  # 空列表
                cleaned[key] = ""
            elif all(isinstance(v, (str, int, float)) for v in value):
                # 简单类型列表，转换为逗号分隔
                cleaned[key] = ", ".join(str(v) for v in value)
            else:
                # 复杂类型列表，转换为JSON
                try:
                    cleaned[key] = json.dumps(value, ensure_ascii=False)
                except:
                    cleaned[key] = str(value)
                    
        # 字典转换为JSON字符串
        elif isinstance(value, dict):
            try:
                cleaned[key] = json.dumps(value, ensure_ascii=False)
            except:
                cleaned[key] = str(value)
                
        # 其他类型转换为字符串
        else:
            cleaned[key] = str(value)
    
    return cleaned


def merge_metadata(base: Dict[str, Any], additional: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个元数据字典，并清理结果
    
    Args:
        base: 基础元数据
        additional: 要添加的元数据
        
    Returns:
        合并并清理后的元数据
    """
    merged = {}
    
    # 先添加基础元数据
    if base:
        merged.update(base)
    
    # 再添加额外元数据（会覆盖同名键）
    if additional:
        merged.update(additional)
    
    # 清理并返回
    return clean_metadata(merged)