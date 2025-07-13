"""Base classes for chunking system."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    """Result of a single chunk."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    
    def __post_init__(self):
        if self.end_char == 0 and self.content:
            self.end_char = self.start_char + len(self.content)


@dataclass
class ChunkingResult:
    """Result of chunking a document."""
    chunks: List[ChunkResult]
    total_chunks: int
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.total_chunks == 0:
            self.total_chunks = len(self.chunks)


class BaseChunker(ABC):
    """Abstract base class for all chunkers."""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200, 
                 min_chunk_size: int = 100):
        """Initialize chunker with configuration."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.validate_config()
        
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> ChunkingResult:
        """Chunk the text into smaller pieces."""
        pass
    
    def validate_config(self) -> bool:
        """Validate chunker configuration."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size must not exceed chunk_size")
        return True