"""Chunking module for KnowledgeCore Engine."""

from .base import BaseChunker, ChunkResult, ChunkingResult
from .markdown_chunker import MarkdownChunker
from .smart_chunker import SmartChunker
from .pipeline import ChunkingPipeline

# 添加新的导入
from .chunk_agent import ChunkAgent, ChunkConfig, Chunk

__all__ = [
    "BaseChunker",
    "ChunkResult", 
    "ChunkingResult",
    "MarkdownChunker",
    "SmartChunker",
    "ChunkingPipeline",
    'ChunkAgent',
    'ChunkConfig', 
    'Chunk'
]