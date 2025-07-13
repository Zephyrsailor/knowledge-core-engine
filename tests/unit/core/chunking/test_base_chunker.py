"""Tests for base chunker interface."""

import pytest
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


class TestChunkResult:
    """Test ChunkResult dataclass."""
    
    def test_chunk_result_creation(self):
        """Test creating a chunk result."""
        chunk = ChunkResult(
            content="This is a test chunk",
            metadata={"type": "text"},
            start_char=0,
            end_char=20
        )
        assert chunk.content == "This is a test chunk"
        assert chunk.metadata == {"type": "text"}
        assert chunk.start_char == 0
        assert chunk.end_char == 20
        
    def test_chunk_result_auto_end_char(self):
        """Test automatic end_char calculation."""
        chunk = ChunkResult(
            content="Hello world",
            start_char=10
        )
        assert chunk.end_char == 21  # 10 + len("Hello world")
        
    def test_chunk_result_empty_metadata(self):
        """Test chunk with empty metadata."""
        chunk = ChunkResult(content="Test")
        assert chunk.metadata == {}
        

class TestChunkingResult:
    """Test ChunkingResult dataclass."""
    
    def test_chunking_result_creation(self):
        """Test creating a chunking result."""
        chunks = [
            ChunkResult(content="Chunk 1"),
            ChunkResult(content="Chunk 2")
        ]
        result = ChunkingResult(
            chunks=chunks,
            total_chunks=2,
            document_metadata={"source": "test.md"}
        )
        assert len(result.chunks) == 2
        assert result.total_chunks == 2
        assert result.document_metadata == {"source": "test.md"}
        
    def test_chunking_result_auto_total(self):
        """Test automatic total_chunks calculation."""
        chunks = [ChunkResult(content=f"Chunk {i}") for i in range(5)]
        result = ChunkingResult(chunks=chunks, total_chunks=0)
        assert result.total_chunks == 5
        

class TestBaseChunker:
    """Test BaseChunker abstract class."""
    
    def test_base_chunker_is_abstract(self):
        """Test that BaseChunker cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseChunker()
            
    def test_base_chunker_requires_chunk_method(self):
        """Test that subclasses must implement chunk method."""
        class IncompleteChunker(BaseChunker):
            pass
            
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteChunker()
            
    def test_base_chunker_valid_subclass(self):
        """Test creating a valid subclass."""
        class SimpleChunker(BaseChunker):
            def chunk(self, text: str, metadata: Dict[str, Any] = None) -> ChunkingResult:
                # Simple implementation for testing
                chunks = [ChunkResult(content=text)]
                return ChunkingResult(chunks=chunks, total_chunks=1)
                
        chunker = SimpleChunker()
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 200
        assert chunker.min_chunk_size == 100
        
    def test_config_validation(self):
        """Test configuration validation."""
        class SimpleChunker(BaseChunker):
            def chunk(self, text: str, metadata: Dict[str, Any] = None) -> ChunkingResult:
                return ChunkingResult(chunks=[], total_chunks=0)
                
        # Valid config
        chunker = SimpleChunker(chunk_size=1000, chunk_overlap=100)
        assert chunker.validate_config()
        
        # Invalid chunk_size
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SimpleChunker(chunk_size=0).validate_config()
            
        # Invalid overlap
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            SimpleChunker(chunk_overlap=-1).validate_config()
            
        # Overlap >= chunk_size
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            SimpleChunker(chunk_size=100, chunk_overlap=100).validate_config()
            
        # Invalid min_chunk_size
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            SimpleChunker(min_chunk_size=0).validate_config()
            
        # min_chunk_size > chunk_size
        # Need to ensure chunk_overlap is valid when testing min_chunk_size
        with pytest.raises(ValueError, match="min_chunk_size must not exceed chunk_size"):
            SimpleChunker(chunk_size=100, chunk_overlap=50, min_chunk_size=200).validate_config()
            
    def test_custom_configuration(self):
        """Test creating chunker with custom configuration."""
        class SimpleChunker(BaseChunker):
            def chunk(self, text: str, metadata: Dict[str, Any] = None) -> ChunkingResult:
                return ChunkingResult(chunks=[], total_chunks=0)
                
        chunker = SimpleChunker(
            chunk_size=2048,
            chunk_overlap=512,
            min_chunk_size=256
        )
        assert chunker.chunk_size == 2048
        assert chunker.chunk_overlap == 512
        assert chunker.min_chunk_size == 256