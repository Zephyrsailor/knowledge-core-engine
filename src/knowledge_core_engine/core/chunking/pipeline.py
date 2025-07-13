"""Chunking pipeline for processing parsed documents."""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

from ..parsing.base import ParseResult
from .base import ChunkingResult, ChunkResult
from .smart_chunker import SmartChunker
from .markdown_chunker import MarkdownChunker

try:
    from llama_index.core.schema import TextNode
except ImportError:
    # Mock for testing
    class TextNode:
        def __init__(self, text: str, metadata: Dict[str, Any] = None, 
                     id_: str = None, relationships: Dict[str, Any] = None):
            self.text = text
            self.metadata = metadata or {}
            self.id_ = id_
            self.relationships = relationships or {}


logger = logging.getLogger(__name__)


class ChunkingPipeline:
    """Pipeline for processing parsed documents into chunks."""
    
    def __init__(self, chunker=None, enable_smart_chunking: bool = True):
        """Initialize the chunking pipeline.
        
        Args:
            chunker: Custom chunker instance. If None, uses default based on enable_smart_chunking
            enable_smart_chunking: Whether to use smart content-aware chunking
        """
        if chunker:
            self.chunker = chunker
        else:
            if enable_smart_chunking:
                self.chunker = SmartChunker()
            else:
                self.chunker = MarkdownChunker()
        
        self.enable_smart_chunking = enable_smart_chunking
        
    async def process_parse_result(self, parse_result: ParseResult) -> ChunkingResult:
        """Process a single parse result into chunks.
        
        Args:
            parse_result: Result from document parsing
            
        Returns:
            ChunkingResult containing all chunks
        """
        try:
            # Extract markdown content
            markdown = parse_result.markdown
            metadata = parse_result.metadata.copy() if parse_result.metadata else {}
            
            # Add parsing metadata if available
            if hasattr(parse_result, 'parsed_at'):
                metadata['parsed_at'] = parse_result.parsed_at
            if hasattr(parse_result, 'parser_used'):
                metadata['parser_used'] = parse_result.parser_used
            if hasattr(parse_result, 'status'):
                metadata['parse_status'] = parse_result.status
            
            # Perform chunking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.chunker.chunk,
                markdown,
                metadata
            )
            
            # Add document-level metadata to all chunks
            for chunk in result.chunks:
                chunk.metadata.update({
                    'source': metadata.get('source', metadata.get('file_path', 'unknown')),
                    'document_id': metadata.get('document_id', None)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            # Return empty result on error
            return ChunkingResult(
                chunks=[],
                total_chunks=0,
                document_metadata=metadata
            )
    
    async def process_batch(self, parse_results: List[ParseResult]) -> List[ChunkingResult]:
        """Process multiple parse results in batch.
        
        Args:
            parse_results: List of parse results
            
        Returns:
            List of chunking results maintaining order
        """
        # Process all documents concurrently
        tasks = [self.process_parse_result(pr) for pr in parse_results]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing document {i}: {result}")
                # Create empty result for failed document
                processed_results.append(ChunkingResult(
                    chunks=[],
                    total_chunks=0,
                    document_metadata={'error': str(result)}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def create_llama_index_nodes(self, chunking_result: ChunkingResult) -> List[TextNode]:
        """Convert chunking result to LlamaIndex TextNode objects.
        
        Args:
            chunking_result: Result from chunking
            
        Returns:
            List of TextNode objects
        """
        nodes = []
        
        # Generate document ID if not present
        doc_id = chunking_result.document_metadata.get(
            'document_id',
            f"doc_{datetime.now().timestamp()}"
        )
        
        for i, chunk in enumerate(chunking_result.chunks):
            # Create unique node ID
            node_id = f"{doc_id}_chunk_{i}"
            
            # Merge chunk and document metadata
            node_metadata = {
                **chunking_result.document_metadata,
                **chunk.metadata,
                'chunk_index': i,
                'total_chunks': chunking_result.total_chunks,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char
            }
            
            # Create relationships - LlamaIndex expects a different format
            relationships = {}
            
            # Create TextNode
            node = TextNode(
                text=chunk.content,
                metadata=node_metadata,
                id_=node_id,
                relationships=relationships
            )
            
            nodes.append(node)
        
        return nodes
    
    def get_chunking_stats(self, chunking_result: ChunkingResult) -> Dict[str, Any]:
        """Get statistics about the chunking result.
        
        Args:
            chunking_result: Result from chunking
            
        Returns:
            Dictionary of statistics
        """
        if not chunking_result.chunks:
            return {
                'total_chunks': 0,
                'average_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_characters': 0
            }
        
        chunk_sizes = [len(chunk.content) for chunk in chunking_result.chunks]
        
        return {
            'total_chunks': chunking_result.total_chunks,
            'average_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'content_types': self._count_content_types(chunking_result)
        }
    
    def _count_content_types(self, chunking_result: ChunkingResult) -> Dict[str, int]:
        """Count chunks by content type."""
        type_counts = {}
        
        for chunk in chunking_result.chunks:
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        return type_counts