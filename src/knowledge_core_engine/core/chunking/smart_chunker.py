"""Smart content-aware chunking implementation with V1 metadata design."""

import re
import hashlib
from typing import List, Dict, Any, Optional
from .base import BaseChunker, ChunkResult, ChunkingResult


class SmartChunker(BaseChunker):
    """Content-aware chunker that adapts to different content types."""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        """Initialize smart chunker."""
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> ChunkingResult:
        """Intelligently chunk text based on content type."""
        if not text:
            return ChunkingResult(chunks=[], total_chunks=0,
                                 document_metadata=metadata or {})
        
        # Generate document_id
        document_id = self._generate_document_id(text, metadata)
        
        # Detect content type
        content_type = self._detect_content_type(text)
        
        # Choose chunking strategy based on content type
        if content_type == "qa":
            chunks = self._chunk_qa_format(text)
        elif content_type == "dialogue":
            chunks = self._chunk_dialogue(text)
        elif content_type == "technical":
            chunks = self._chunk_technical(text)
        else:
            chunks = self._chunk_generic(text)
        
        # Apply V1 metadata to all chunks
        source_path = metadata.get('source', '') if metadata else ''
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            
            # Build hierarchy path based on content structure
            hierarchy_path = self._build_hierarchy_path(chunk, content_type, i)
            
            # Determine parent chunk
            parent_chunk_id = self._determine_parent_chunk(
                chunks, i, document_id, content_type
            )
            
            # Update with V1 metadata
            v1_metadata = {
                'chunk_id': chunk_id,
                'document_id': document_id,
                'source_path': source_path,
                'hierarchy_path': hierarchy_path,
                'parent_chunk_id': parent_chunk_id,
                'chunk_index': i,
                'content_type': chunk.metadata.get('chunk_type', content_type)
            }
            
            # Preserve any existing metadata
            chunk.metadata.update(v1_metadata)
            
            # Add detected entities if not already present
            if 'detected_entities' not in chunk.metadata:
                chunk.metadata['detected_entities'] = self._extract_entities(chunk.content)
        
        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            document_metadata={
                'document_id': document_id,
                'source_path': source_path,
                'content_type': content_type,
                'chunking_method': 'smart',
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
    
    def _build_hierarchy_path(self, chunk: ChunkResult, content_type: str, 
                             index: int) -> str:
        """Build hierarchy path based on content structure."""
        chunk_type = chunk.metadata.get('chunk_type', '')
        
        if content_type == "qa":
            if chunk_type == 'qa_pair':
                # Extract question as hierarchy
                content = chunk.content
                q_match = re.search(r'(?:Q:|Question:|问：)\s*(.+?)(?:\n|$)', content)
                if q_match:
                    return f"QA/{q_match.group(1).strip()[:50]}"
            return f"QA/Item_{index}"
            
        elif content_type == "dialogue":
            turn_index = chunk.metadata.get('turn_index', index)
            return f"Dialogue/Turn_{turn_index}"
            
        elif content_type == "technical":
            if chunk_type == 'code_block':
                # Try to extract function/class name
                code_match = re.search(r'(?:def|class|function)\s+(\w+)', chunk.content)
                if code_match:
                    return f"Code/{code_match.group(1)}"
                return f"Code/Block_{index}"
            return f"Technical/Section_{index}"
            
        else:  # narrative
            # For narrative, use position-based hierarchy
            section_num = index // 5  # Group every 5 chunks
            return f"Content/Section_{section_num}/Chunk_{index}"
    
    def _determine_parent_chunk(self, chunks: List[ChunkResult], current_index: int,
                               document_id: str, content_type: str) -> Optional[str]:
        """Determine parent chunk based on content structure."""
        if current_index == 0:
            return None
        
        chunk_type = chunks[current_index].metadata.get('chunk_type', '')
        
        # For Q&A, each pair is independent
        if content_type == "qa" and chunk_type in ['qa_pair', 'question_only']:
            return None
        
        # For dialogue, link consecutive turns
        if content_type == "dialogue":
            return f"{document_id}_chunk_{current_index - 1}"
        
        # For technical content, link text to previous text, code to previous
        if content_type == "technical":
            # Find the previous non-code chunk if current is code
            if chunk_type == 'code_block':
                for i in range(current_index - 1, -1, -1):
                    if chunks[i].metadata.get('chunk_type') != 'code_block':
                        return f"{document_id}_chunk_{i}"
            else:
                # Link to previous chunk
                return f"{document_id}_chunk_{current_index - 1}"
        
        # For narrative, simple sequential linking
        return f"{document_id}_chunk_{current_index - 1}"
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content."""
        # Simple heuristics for content detection
        qa_pattern = r'(Q:|Question:|A:|Answer:|问：|答：)'
        dialogue_pattern = r'(Speaker \d+:|说话人\d+：|^-\s*".*"$)'
        code_pattern = r'(```|class\s+\w+|def\s+\w+|function\s+\w+|import\s+\w+)'
        
        if re.search(qa_pattern, text, re.MULTILINE):
            return "qa"
        elif re.search(dialogue_pattern, text, re.MULTILINE):
            return "dialogue"
        elif re.search(code_pattern, text, re.MULTILINE):
            return "technical"
        else:
            return "narrative"
    
    def _chunk_qa_format(self, text: str) -> List[ChunkResult]:
        """Chunk Q&A formatted text."""
        chunks = []
        
        # Split by Q&A pairs
        qa_pattern = r'((?:Q:|Question:|问：).*?)(?=(?:Q:|Question:|问：)|$)'
        matches = re.findall(qa_pattern, text, re.DOTALL)
        
        current_pos = 0
        for match in matches:
            # Find the actual position in the text
            match_start = text.find(match, current_pos)
            match_end = match_start + len(match)
            
            # Split Q and A if possible
            answer_pattern = r'(A:|Answer:|答：)'
            answer_match = re.search(answer_pattern, match)
            
            if answer_match:
                q_content = match[:answer_match.start()].strip()
                a_content = match[answer_match.start():].strip()
                
                # Create chunk with Q&A pair
                chunk_content = f"{q_content}\n{a_content}"
                chunks.append(ChunkResult(
                    content=chunk_content,
                    metadata={
                        'chunk_type': 'qa_pair',
                        'has_question': True,
                        'has_answer': True
                    },
                    start_char=match_start,
                    end_char=match_end
                ))
            else:
                # Just a question without answer
                chunks.append(ChunkResult(
                    content=match.strip(),
                    metadata={
                        'chunk_type': 'question_only',
                        'has_question': True,
                        'has_answer': False
                    },
                    start_char=match_start,
                    end_char=match_end
                ))
            
            current_pos = match_end
        
        # Handle any remaining text
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                chunks.append(ChunkResult(
                    content=remaining,
                    metadata={'chunk_type': 'other'},
                    start_char=current_pos,
                    end_char=len(text)
                ))
        
        return chunks if chunks else self._chunk_generic(text)
    
    def _chunk_dialogue(self, text: str) -> List[ChunkResult]:
        """Chunk dialogue/conversation text."""
        chunks = []
        
        # Split by speaker turns
        speaker_pattern = r'((?:Speaker \d+:|说话人\d+：|^-\s*").*?)(?=(?:Speaker \d+:|说话人\d+：|^-\s*")|$)'
        matches = re.findall(speaker_pattern, text, re.MULTILINE | re.DOTALL)
        
        if not matches:
            return self._chunk_generic(text)
        
        current_pos = 0
        for i, match in enumerate(matches):
            match_start = text.find(match, current_pos)
            match_end = match_start + len(match)
            
            chunks.append(ChunkResult(
                content=match.strip(),
                metadata={
                    'chunk_type': 'dialogue_turn',
                    'turn_index': i
                },
                start_char=match_start,
                end_char=match_end
            ))
            
            current_pos = match_end
        
        return chunks
    
    def _chunk_technical(self, text: str) -> List[ChunkResult]:
        """Chunk technical documentation."""
        chunks = []
        
        # Protect code blocks
        code_blocks = []
        code_pattern = r'```[\s\S]*?```'
        
        # Extract code blocks first
        for match in re.finditer(code_pattern, text):
            code_blocks.append({
                'content': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Chunk non-code parts
        current_pos = 0
        for code_block in code_blocks:
            # Process text before code block
            if current_pos < code_block['start']:
                text_chunk = text[current_pos:code_block['start']]
                text_chunks = self._chunk_generic(text_chunk)
                
                # Adjust positions
                for chunk in text_chunks:
                    chunk.start_char += current_pos
                    chunk.end_char += current_pos
                    chunk.metadata['chunk_type'] = 'technical_text'
                
                chunks.extend(text_chunks)
            
            # Add code block as single chunk
            chunks.append(ChunkResult(
                content=code_block['content'],
                metadata={
                    'chunk_type': 'code_block',
                    'is_code': True
                },
                start_char=code_block['start'],
                end_char=code_block['end']
            ))
            
            current_pos = code_block['end']
        
        # Process remaining text
        if current_pos < len(text):
            remaining = text[current_pos:]
            remaining_chunks = self._chunk_generic(remaining)
            
            for chunk in remaining_chunks:
                chunk.start_char += current_pos
                chunk.end_char += current_pos
                chunk.metadata['chunk_type'] = 'technical_text'
            
            chunks.extend(remaining_chunks)
        
        return chunks if chunks else self._chunk_generic(text)
    
    def _chunk_generic(self, text: str) -> List[ChunkResult]:
        """Generic chunking for narrative text."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_size = 0
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If paragraph itself is too large
            if para_size > self.chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    chunks.append(ChunkResult(
                        content=chunk_content,
                        metadata={'chunk_type': 'narrative'},
                        start_char=current_start,
                        end_char=current_start + len(chunk_content)
                    ))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if len(sent) <= self.chunk_size:
                        chunks.append(ChunkResult(
                            content=sent,
                            metadata={'chunk_type': 'narrative'},
                            start_char=current_start,
                            end_char=current_start + len(sent)
                        ))
                        current_start += len(sent) + 1
                    else:
                        # Split very long sentence
                        for i in range(0, len(sent), self.chunk_size):
                            chunk_text = sent[i:i+self.chunk_size]
                            chunks.append(ChunkResult(
                                content=chunk_text,
                                metadata={'chunk_type': 'narrative'},
                                start_char=current_start + i,
                                end_char=current_start + i + len(chunk_text)
                            ))
            
            # If adding this paragraph would exceed chunk size
            elif current_size + para_size + 2 > self.chunk_size:  # +2 for \n\n
                # Save current chunk
                if current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    chunks.append(ChunkResult(
                        content=chunk_content,
                        metadata={'chunk_type': 'narrative'},
                        start_char=current_start,
                        end_char=current_start + len(chunk_content)
                    ))
                
                # Start new chunk
                current_chunk = [para]
                current_size = para_size
                current_start = current_start + len(chunk_content) + 2
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_size += para_size + 2
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append(ChunkResult(
                content=chunk_content,
                metadata={'chunk_type': 'narrative'},
                start_char=current_start,
                end_char=current_start + len(chunk_content)
            ))
        
        return chunks
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract important entities from text."""
        entities = []
        
        # Simple pattern matching for now
        # Extract quoted strings
        quotes = re.findall(r'"([^"]+)"', text)
        entities.extend(quotes[:3])  # Limit to avoid too many
        
        # Extract capitalized words (potential names)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(caps[:3])
        
        # Extract technical terms (CamelCase)
        camel = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        entities.extend(camel[:3])
        
        return list(set(entities))  # Remove duplicates