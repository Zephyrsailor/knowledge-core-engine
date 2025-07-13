"""Markdown-specific chunking implementation with V1 metadata design."""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseChunker, ChunkResult, ChunkingResult


class MarkdownChunker(BaseChunker):
    """Chunker optimized for Markdown documents with hierarchical metadata."""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        """Initialize Markdown chunker."""
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> ChunkingResult:
        """Chunk Markdown text by headers while preserving structure."""
        if not text:
            return ChunkingResult(chunks=[], total_chunks=0, 
                                 document_metadata=metadata or {})
        
        # Generate document_id
        document_id = self._generate_document_id(text, metadata)
        
        # Split by headers while preserving hierarchy
        sections = self._split_by_headers(text)
        
        # Build hierarchy structure
        hierarchy_tree = self._build_hierarchy_tree(sections)
        
        # Process sections into chunks
        chunks = []
        chunk_index = 0
        source_path = metadata.get('source', '') if metadata else ''
        
        for section in sections:
            # Add source_path to section metadata
            section['source_path'] = source_path
            section_chunks = self._process_section_v1(
                section, 
                document_id, 
                chunk_index,
                hierarchy_tree
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        # Apply overlap if configured
        if chunks and self.chunk_overlap > 0:
            chunks = self._apply_overlap_v1(chunks)
        
        # Set parent_chunk_id relationships based on hierarchy
        self._set_parent_relationships(chunks)
        
        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            document_metadata={
                'document_id': document_id,
                'source_path': metadata.get('source', '') if metadata else '',
                'chunking_method': 'markdown_structure',
                **(metadata or {})
            }
        )
    
    def _generate_document_id(self, text: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Generate a unique document ID."""
        if metadata and metadata.get('document_id'):
            return metadata['document_id']
        
        # Use source path if available
        if metadata and metadata.get('source'):
            source_hash = hashlib.md5(str(metadata['source']).encode()).hexdigest()[:8]
            return f"doc_{source_hash}"
        
        # Fallback to content hash
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"doc_{content_hash}"
    
    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """Split text by Markdown headers while tracking hierarchy."""
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        sections = []
        current_headers = []  # Stack of headers for hierarchy
        current_section = {
            'headers': [],
            'header_levels': [],
            'content': [],
            'start_pos': 0
        }
        
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            
            if header_match:
                # Save previous section if it has content
                if current_section['content']:
                    content_text = '\n'.join(current_section['content'])
                    if content_text.strip():
                        sections.append({
                            'headers': current_section['headers'].copy(),
                            'header_levels': current_section['header_levels'].copy(),
                            'content': content_text,
                            'start_pos': current_section['start_pos'],
                            'end_pos': current_pos - 1
                        })
                
                # Update header hierarchy
                header_level = len(header_match.group(1))
                header_text = header_match.group(2)
                
                # Pop headers until we find the right level
                while (current_headers and 
                       current_headers[-1]['level'] >= header_level):
                    current_headers.pop()
                
                # Add current header
                current_headers.append({
                    'level': header_level,
                    'text': header_text
                })
                
                # Start new section
                current_section = {
                    'headers': [h['text'] for h in current_headers],
                    'header_levels': [h['level'] for h in current_headers],
                    'content': [line],  # Include header in content
                    'start_pos': current_pos
                }
            else:
                current_section['content'].append(line)
            
            current_pos += len(line) + 1  # +1 for newline
        
        # Don't forget the last section
        if current_section['content']:
            content_text = '\n'.join(current_section['content'])
            if content_text.strip():
                sections.append({
                    'headers': current_section['headers'].copy(),
                    'header_levels': current_section['header_levels'].copy(),
                    'content': content_text,
                    'start_pos': current_section['start_pos'],
                    'end_pos': current_pos
                })
        
        return sections
    
    def _build_hierarchy_tree(self, sections: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Build a tree structure to track parent-child relationships."""
        hierarchy_tree = {}
        
        for i, section in enumerate(sections):
            # Find parent section
            parent_idx = None
            if i > 0 and section['header_levels']:
                current_level = section['header_levels'][-1]
                # Look backwards for a section with lower level
                for j in range(i - 1, -1, -1):
                    if (sections[j]['header_levels'] and 
                        sections[j]['header_levels'][-1] < current_level):
                        parent_idx = j
                        break
            
            hierarchy_tree[i] = {
                'parent': parent_idx,
                'children': []
            }
            
            # Update parent's children list
            if parent_idx is not None:
                hierarchy_tree[parent_idx]['children'].append(i)
        
        return hierarchy_tree
    
    def _process_section_v1(self, section: Dict[str, Any], document_id: str,
                           start_chunk_index: int, 
                           hierarchy_tree: Dict[int, Dict]) -> List[ChunkResult]:
        """Process a section into chunks with V1 metadata."""
        content = section['content']
        
        # Check for special content
        is_code = self._is_code_block(content)
        is_table = self._is_table(content)
        
        # For special content, try to keep as single chunk
        if (is_code or is_table) and len(content) <= self.chunk_size * 1.5:
            return [self._create_chunk_v1(
                content=content,
                document_id=document_id,
                chunk_index=start_chunk_index,
                section=section,
                is_code=is_code,
                is_table=is_table
            )]
        
        # Otherwise, chunk by size
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # Find good break point
            if end < len(content):
                # Look for paragraph break first
                last_double_newline = content.rfind('\n\n', start, end)
                if last_double_newline > start + self.min_chunk_size:
                    end = last_double_newline + 2
                else:
                    # Look for sentence end
                    last_period = content.rfind('. ', start, end)
                    if last_period > start + self.min_chunk_size:
                        end = last_period + 2
                    else:
                        # Look for any newline
                        last_newline = content.rfind('\n', start, end)
                        if last_newline > start + self.min_chunk_size:
                            end = last_newline + 1
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunks.append(self._create_chunk_v1(
                    content=chunk_text,
                    document_id=document_id,
                    chunk_index=start_chunk_index + len(chunks),
                    section=section,
                    start_offset=start,
                    end_offset=end
                ))
            
            start = end
        
        return chunks
    
    def _create_chunk_v1(self, content: str, document_id: str, chunk_index: int,
                        section: Dict[str, Any], is_code: bool = False,
                        is_table: bool = False, start_offset: int = 0,
                        end_offset: Optional[int] = None) -> ChunkResult:
        """Create a chunk with V1 metadata."""
        # Generate chunk_id
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        
        # Build hierarchy_path
        hierarchy_path = "/".join(section['headers']) if section['headers'] else ""
        
        # Determine content_type
        if is_code:
            content_type = "code"
        elif is_table:
            content_type = "table"
        elif section['headers'] and any(
            'example' in h.lower() or 'code' in h.lower() 
            for h in section['headers']
        ):
            content_type = "example"
        else:
            content_type = "text"
        
        # V1 metadata
        metadata = {
            'chunk_id': chunk_id,
            'document_id': document_id,
            'source_path': section.get('source_path', ''),  # Get from section metadata
            'hierarchy_path': hierarchy_path,
            'parent_chunk_id': None,  # Will be set later
            'chunk_index': chunk_index,
            'content_type': content_type
        }
        
        # Calculate character positions
        if end_offset is None:
            end_offset = len(content)
        
        start_char = section['start_pos'] + start_offset
        end_char = section['start_pos'] + end_offset
        
        return ChunkResult(
            content=content,
            metadata=metadata,
            start_char=start_char,
            end_char=end_char
        )
    
    def _apply_overlap_v1(self, chunks: List[ChunkResult]) -> List[ChunkResult]:
        """Apply overlap while preserving V1 metadata."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk.content[-self.chunk_overlap:]
                
                # Create new chunk with overlap
                new_content = overlap_text + " " + chunk.content
                new_metadata = chunk.metadata.copy()
                new_metadata['has_overlap'] = True
                new_metadata['overlap_size'] = len(overlap_text)
                
                new_chunk = ChunkResult(
                    content=new_content,
                    metadata=new_metadata,
                    start_char=chunk.start_char - len(overlap_text) - 1,
                    end_char=chunk.end_char
                )
                overlapped_chunks.append(new_chunk)
        
        return overlapped_chunks
    
    def _set_parent_relationships(self, chunks: List[ChunkResult]) -> None:
        """Set parent_chunk_id based on hierarchy paths."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Look for parent based on hierarchy_path
                current_path = chunk.metadata['hierarchy_path']
                current_parts = current_path.split('/') if current_path else []
                
                # Find the most recent chunk with shorter hierarchy path
                for j in range(i - 1, -1, -1):
                    prev_path = chunks[j].metadata['hierarchy_path']
                    prev_parts = prev_path.split('/') if prev_path else []
                    
                    # Check if prev is parent (shorter path that is prefix of current)
                    if (len(prev_parts) < len(current_parts) and
                        current_parts[:len(prev_parts)] == prev_parts):
                        chunk.metadata['parent_chunk_id'] = chunks[j].metadata['chunk_id']
                        break
    
    def _is_code_block(self, text: str) -> bool:
        """Check if text is a code block."""
        lines = text.strip().split('\n')
        if len(lines) >= 2:
            return lines[0].startswith('```') and lines[-1].strip() == '```'
        return False
    
    def _is_table(self, text: str) -> bool:
        """Check if text contains a Markdown table."""
        lines = text.strip().split('\n')
        if len(lines) >= 2:
            # Check for table separator line
            for line in lines:
                if re.match(r'^\s*\|?\s*:?-+:?\s*\|', line):
                    return True
        return False