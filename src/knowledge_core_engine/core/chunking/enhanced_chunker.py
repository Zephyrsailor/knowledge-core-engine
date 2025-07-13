"""Enhanced chunker with advanced metadata for complex retrieval scenarios."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
from collections import defaultdict

from .base import BaseChunker, ChunkResult, ChunkingResult


@dataclass
class HierarchyMetadata:
    """Metadata for hierarchical relationships."""
    chunk_id: str
    parent_chunk_id: Optional[str] = None
    root_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    hierarchy_path: List[str] = field(default_factory=list)
    hierarchy_levels: List[int] = field(default_factory=list)
    sibling_chunks: List[str] = field(default_factory=list)
    position_in_parent: int = 0


@dataclass
class ContextMetadata:
    """Metadata for context window and multi-hop retrieval."""
    prev_chunks: List[str] = field(default_factory=list)
    next_chunks: List[str] = field(default_factory=list)
    context_summary: Optional[str] = None
    section_summary: Optional[str] = None
    references_to: List[str] = field(default_factory=list)
    referenced_by: List[str] = field(default_factory=list)


@dataclass
class SemanticMetadata:
    """Metadata for semantic relationships and concept graph."""
    key_concepts: List[str] = field(default_factory=list)
    defined_terms: List[str] = field(default_factory=list)
    used_terms: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    leads_to: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    topic_importance: Dict[str, float] = field(default_factory=dict)


class EnhancedChunker(BaseChunker):
    """Enhanced chunker with rich metadata for advanced retrieval."""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200,
                 min_chunk_size: int = 100, context_window: int = 2):
        """Initialize enhanced chunker.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            context_window: Number of chunks to include in context
        """
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.context_window = context_window
        
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> ChunkingResult:
        """Chunk text with enhanced metadata."""
        if not text:
            return ChunkingResult(chunks=[], total_chunks=0,
                                 document_metadata=metadata or {})
        
        # First, create basic chunks with hierarchical structure
        hierarchical_chunks = self._create_hierarchical_chunks(text)
        
        # Build chunk index for cross-references
        chunk_index = {chunk['id']: chunk for chunk in hierarchical_chunks}
        
        # Enhance with context and semantic metadata
        enhanced_chunks = []
        for i, chunk_data in enumerate(hierarchical_chunks):
            # Create context metadata
            context_meta = self._build_context_metadata(
                i, hierarchical_chunks, self.context_window
            )
            
            # Extract semantic information
            semantic_meta = self._extract_semantic_metadata(
                chunk_data['content'], chunk_index
            )
            
            # Build hierarchy metadata
            hierarchy_meta = self._build_hierarchy_metadata(
                chunk_data, hierarchical_chunks
            )
            
            # Combine all metadata
            full_metadata = {
                **chunk_data.get('base_metadata', {}),
                'hierarchy': hierarchy_meta,
                'context': context_meta,
                'semantics': semantic_meta,
                'chunk_id': chunk_data['id'],
                'chunk_index': i,
                'importance_score': self._calculate_importance(
                    chunk_data, hierarchy_meta, semantic_meta
                )
            }
            
            enhanced_chunks.append(ChunkResult(
                content=chunk_data['content'],
                metadata=full_metadata,
                start_char=chunk_data['start'],
                end_char=chunk_data['end']
            ))
        
        # Post-process to add cross-references
        self._add_cross_references(enhanced_chunks)
        
        return ChunkingResult(
            chunks=enhanced_chunks,
            total_chunks=len(enhanced_chunks),
            document_metadata={**(metadata or {}), 'enhanced': True}
        )
    
    def _create_hierarchical_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create chunks with hierarchical structure based on Markdown headers."""
        chunks = []
        
        # Split by headers while preserving hierarchy
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        current_hierarchy = []
        current_content = []
        current_start = 0
        
        lines = text.split('\n')
        pos = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            
            if header_match:
                # Save previous chunk if exists
                if current_content:
                    chunk_text = '\n'.join(current_content)
                    if chunk_text.strip():
                        chunk_id = self._generate_chunk_id(chunk_text, len(chunks))
                        chunks.append({
                            'id': chunk_id,
                            'content': chunk_text,
                            'hierarchy': current_hierarchy.copy(),
                            'start': current_start,
                            'end': pos - 1,
                            'base_metadata': {
                                'headers': [h['title'] for h in current_hierarchy]
                            }
                        })
                
                # Update hierarchy
                level = len(header_match.group(1))
                title = header_match.group(2)
                
                # Remove deeper levels
                while current_hierarchy and current_hierarchy[-1]['level'] >= level:
                    current_hierarchy.pop()
                
                # Add current header
                current_hierarchy.append({
                    'level': level,
                    'title': title
                })
                
                current_content = [line]
                current_start = pos
            else:
                current_content.append(line)
            
            pos += len(line) + 1
        
        # Don't forget last chunk
        if current_content:
            chunk_text = '\n'.join(current_content)
            if chunk_text.strip():
                chunk_id = self._generate_chunk_id(chunk_text, len(chunks))
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_text,
                    'hierarchy': current_hierarchy.copy(),
                    'start': current_start,
                    'end': pos,
                    'base_metadata': {
                        'headers': [h['title'] for h in current_hierarchy]
                    }
                })
        
        # Now split large chunks if needed
        final_chunks = []
        for chunk in chunks:
            if len(chunk['content']) > self.chunk_size:
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a large chunk into smaller ones while preserving hierarchy."""
        sub_chunks = []
        content = chunk['content']
        
        # Split by paragraphs or sentences
        parts = self._smart_split(content, self.chunk_size)
        
        for i, part in enumerate(parts):
            sub_id = f"{chunk['id']}_sub{i}"
            sub_chunks.append({
                'id': sub_id,
                'content': part,
                'hierarchy': chunk['hierarchy'],
                'start': chunk['start'] + sum(len(p) for p in parts[:i]),
                'end': chunk['start'] + sum(len(p) for p in parts[:i+1]),
                'base_metadata': {
                    **chunk['base_metadata'],
                    'is_sub_chunk': True,
                    'parent_chunk': chunk['id'],
                    'sub_index': i
                }
            })
        
        return sub_chunks
    
    def _smart_split(self, text: str, max_size: int) -> List[str]:
        """Smart splitting that respects sentence and paragraph boundaries."""
        if len(text) <= max_size:
            return [text]
        
        # Try splitting by paragraphs first
        paragraphs = text.split('\n\n')
        
        result = []
        current = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > max_size and current:
                result.append('\n\n'.join(current))
                current = [para]
                current_size = para_size
            else:
                current.append(para)
                current_size += para_size + 2  # Account for \n\n
        
        if current:
            result.append('\n\n'.join(current))
        
        # Further split if any chunk is still too large
        final_result = []
        for chunk in result:
            if len(chunk) > max_size:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                sub_chunk = []
                sub_size = 0
                
                for sent in sentences:
                    if sub_size + len(sent) > max_size and sub_chunk:
                        final_result.append(' '.join(sub_chunk))
                        sub_chunk = [sent]
                        sub_size = len(sent)
                    else:
                        sub_chunk.append(sent)
                        sub_size += len(sent) + 1
                
                if sub_chunk:
                    final_result.append(' '.join(sub_chunk))
            else:
                final_result.append(chunk)
        
        return final_result
    
    def _build_hierarchy_metadata(self, chunk_data: Dict[str, Any],
                                  all_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build hierarchy metadata for a chunk."""
        chunk_id = chunk_data['id']
        hierarchy = chunk_data['hierarchy']
        
        # Find parent chunk
        parent_id = None
        if hierarchy:
            # Parent is the chunk with hierarchy depth one less
            parent_hierarchy = hierarchy[:-1]
            for other in all_chunks:
                if (other['hierarchy'] == parent_hierarchy and 
                    other['id'] != chunk_id):
                    parent_id = other['id']
                    break
        
        # Find children
        child_ids = []
        for other in all_chunks:
            other_hierarchy = other['hierarchy']
            if (len(other_hierarchy) == len(hierarchy) + 1 and
                other_hierarchy[:-1] == hierarchy):
                child_ids.append(other['id'])
        
        # Find siblings
        sibling_ids = []
        for other in all_chunks:
            if (other['hierarchy'] == hierarchy and 
                other['id'] != chunk_id):
                sibling_ids.append(other['id'])
        
        return {
            'chunk_id': chunk_id,
            'parent_chunk_id': parent_id,
            'child_chunk_ids': child_ids,
            'sibling_chunks': sibling_ids,
            'hierarchy_path': [h['title'] for h in hierarchy],
            'hierarchy_levels': [h['level'] for h in hierarchy],
            'depth': len(hierarchy)
        }
    
    def _build_context_metadata(self, index: int, all_chunks: List[Dict[str, Any]],
                               window: int) -> Dict[str, Any]:
        """Build context window metadata."""
        prev_chunks = []
        next_chunks = []
        
        # Previous chunks
        for i in range(max(0, index - window), index):
            prev_chunks.append(all_chunks[i]['id'])
        
        # Next chunks
        for i in range(index + 1, min(len(all_chunks), index + window + 1)):
            next_chunks.append(all_chunks[i]['id'])
        
        return {
            'prev_chunks': prev_chunks,
            'next_chunks': next_chunks,
            'window_size': window
        }
    
    def _extract_semantic_metadata(self, content: str,
                                  chunk_index: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic information from chunk content."""
        # Extract key concepts (capitalized phrases, technical terms)
        concepts = []
        
        # Find capitalized phrases
        cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        concepts.extend(re.findall(cap_pattern, content))
        
        # Find technical terms (CamelCase)
        camel_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
        concepts.extend(re.findall(camel_pattern, content))
        
        # Find terms in quotes
        quote_pattern = r'"([^"]+)"'
        concepts.extend(re.findall(quote_pattern, content))
        
        # Detect definitions
        defined_terms = []
        definition_patterns = [
            r'(\w+)\s+is\s+(?:a|an|the)\s+',
            r'(\w+)\s+refers\s+to\s+',
            r'define\s+(\w+)\s+as\s+'
        ]
        
        for pattern in definition_patterns:
            defined_terms.extend(re.findall(pattern, content, re.IGNORECASE))
        
        return {
            'key_concepts': list(set(concepts))[:10],  # Top 10 unique concepts
            'defined_terms': list(set(defined_terms)),
            'has_code': bool(re.search(r'```|def\s+|class\s+|function\s+', content)),
            'has_formula': bool(re.search(r'\$.*\$|\\\[.*\\\]', content)),
            'has_list': bool(re.search(r'^\s*[-*\d]+\.\s+', content, re.MULTILINE))
        }
    
    def _calculate_importance(self, chunk_data: Dict[str, Any],
                            hierarchy: Dict[str, Any],
                            semantics: Dict[str, Any]) -> float:
        """Calculate importance score for a chunk."""
        score = 0.5  # Base score
        
        # Hierarchy factors
        if hierarchy['depth'] == 1:  # Top-level section
            score += 0.2
        elif hierarchy['depth'] == 2:  # Subsection
            score += 0.1
        
        # Content factors
        if semantics.get('defined_terms'):
            score += 0.15
        if semantics.get('key_concepts'):
            score += 0.1
        if semantics.get('has_code'):
            score += 0.05
        
        # Position factors (first and last chunks often important)
        if 'introduction' in chunk_data['content'].lower()[:100]:
            score += 0.1
        if 'conclusion' in chunk_data['content'].lower()[:100]:
            score += 0.1
        
        return min(score, 1.0)
    
    def _add_cross_references(self, chunks: List[ChunkResult]):
        """Add cross-references between chunks based on shared concepts."""
        # Build concept index
        concept_index = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            concepts = chunk.metadata.get('semantics', {}).get('key_concepts', [])
            for concept in concepts:
                concept_index[concept.lower()].append(i)
        
        # Add references based on shared concepts
        for i, chunk in enumerate(chunks):
            references = set()
            concepts = chunk.metadata.get('semantics', {}).get('key_concepts', [])
            
            for concept in concepts:
                # Find other chunks with same concept
                for other_idx in concept_index[concept.lower()]:
                    if other_idx != i:
                        references.add(chunks[other_idx].metadata['chunk_id'])
            
            if references:
                chunk.metadata['semantics']['related_chunks'] = list(references)
    
    def _generate_chunk_id(self, content: str, index: int) -> str:
        """Generate a unique ID for a chunk."""
        # Use content hash + index for uniqueness
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"chunk_{index}_{content_hash}"