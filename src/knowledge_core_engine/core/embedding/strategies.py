"""Embedding strategies for different use cases.

This module provides various strategies for preparing text before embedding:
- SimpleStrategy: Just embed the raw content
- MultiVectorStrategy: Combine content + summary + questions
- HybridStrategy: Custom weighted combination
"""

from typing import Dict, Any, Optional, List
from abc import abstractmethod
import json

from .base import IEmbeddingStrategy


class EmbeddingStrategy(IEmbeddingStrategy):
    """Base class for embedding strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy with configuration."""
        self.config = config or {}
    
    @abstractmethod
    def prepare_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare text for embedding."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass


class SimpleStrategy(EmbeddingStrategy):
    """Simple strategy that only embeds the raw content."""
    
    def prepare_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Return the content as-is.
        
        Args:
            content: Original text content
            metadata: Ignored in simple strategy
            
        Returns:
            Original content
        """
        return content.strip()
    
    def get_strategy_name(self) -> str:
        return "simple"


class MultiVectorStrategy(EmbeddingStrategy):
    """Multi-vector strategy combining content + summary + questions.
    
    This strategy enriches the embedding by including:
    - Original content
    - AI-generated summary
    - Potential questions that might lead to this content
    
    This helps improve retrieval accuracy by capturing different
    semantic aspects of the content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration.
        
        Config options:
            include_summary (bool): Include summary in embedding
            include_questions (bool): Include questions in embedding
            include_keywords (bool): Include keywords in embedding
            separator (str): Text separator between sections
        """
        super().__init__(config)
        self.include_summary = self.config.get("include_summary", True)
        self.include_questions = self.config.get("include_questions", True)
        self.include_keywords = self.config.get("include_keywords", False)
        self.separator = self.config.get("separator", "\n\n")
    
    def prepare_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Combine content with metadata for richer embedding.
        
        Args:
            content: Original text content
            metadata: Should contain 'summary', 'questions', etc.
            
        Returns:
            Combined text for embedding
        """
        parts = []
        
        # Always include original content
        parts.append(f"Content: {content.strip()}")
        
        # Add summary if available and enabled
        if self.include_summary and metadata.get("summary"):
            parts.append(f"Summary: {metadata['summary']}")
        
        # Add questions if available and enabled
        if self.include_questions and metadata.get("questions"):
            questions = metadata["questions"]
            if isinstance(questions, list):
                questions_text = " ".join(questions)
            else:
                questions_text = str(questions)
            parts.append(f"Questions: {questions_text}")
        
        # Add keywords if available and enabled
        if self.include_keywords and metadata.get("keywords"):
            keywords = metadata["keywords"]
            if isinstance(keywords, list):
                keywords_text = ", ".join(keywords)
            else:
                keywords_text = str(keywords)
            parts.append(f"Keywords: {keywords_text}")
        
        return self.separator.join(parts)
    
    def get_strategy_name(self) -> str:
        return "multi_vector"


class HybridStrategy(EmbeddingStrategy):
    """Hybrid strategy with weighted combination of components.
    
    This strategy allows fine-tuned control over how different
    components contribute to the final embedding by repeating
    them based on weights.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration.
        
        Config options:
            content_weight (float): Weight for original content (default: 0.5)
            summary_weight (float): Weight for summary (default: 0.3)
            questions_weight (float): Weight for questions (default: 0.2)
            normalize_weights (bool): Normalize weights to sum to 1.0
        """
        super().__init__(config)
        self.content_weight = self.config.get("content_weight", 0.5)
        self.summary_weight = self.config.get("summary_weight", 0.3)
        self.questions_weight = self.config.get("questions_weight", 0.2)
        self.normalize_weights = self.config.get("normalize_weights", True)
        
        if self.normalize_weights:
            self._normalize()
    
    def _normalize(self):
        """Normalize weights to sum to 1.0."""
        total = self.content_weight + self.summary_weight + self.questions_weight
        if total > 0:
            self.content_weight /= total
            self.summary_weight /= total
            self.questions_weight /= total
    
    def prepare_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare text with weighted components.
        
        The weighting is implemented by controlling the order and
        emphasis of different components in the final text.
        """
        components = []
        
        # Add content based on weight
        if self.content_weight > 0:
            components.append((self.content_weight, content.strip()))
        
        # Add summary based on weight
        if self.summary_weight > 0 and metadata.get("summary"):
            components.append((self.summary_weight, metadata["summary"]))
        
        # Add questions based on weight
        if self.questions_weight > 0 and metadata.get("questions"):
            questions = metadata.get("questions", [])
            if isinstance(questions, list) and questions:
                questions_text = " ".join(questions)
                components.append((self.questions_weight, questions_text))
        
        # Sort by weight (highest first) for emphasis
        components.sort(key=lambda x: x[0], reverse=True)
        
        # Combine with emphasis markers
        result_parts = []
        for weight, text in components:
            if weight >= 0.5:
                # High weight: add emphasis
                result_parts.append(f"[IMPORTANT] {text}")
            elif weight >= 0.3:
                # Medium weight: normal
                result_parts.append(text)
            else:
                # Low weight: de-emphasize
                result_parts.append(f"[Additional context: {text}]")
        
        return "\n\n".join(result_parts)
    
    def get_strategy_name(self) -> str:
        return "hybrid"


class StructuredStrategy(EmbeddingStrategy):
    """Strategy that preserves document structure in embeddings.
    
    Useful for technical documents, code, or hierarchical content
    where structure carries semantic meaning.
    """
    
    def prepare_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare text while preserving structure.
        
        Args:
            content: Original content
            metadata: Should contain 'hierarchy_path', 'content_type', etc.
            
        Returns:
            Structured text representation
        """
        parts = []
        
        # Add hierarchy context if available
        if metadata.get("hierarchy_path"):
            parts.append(f"Path: {metadata['hierarchy_path']}")
        
        # Add content type context
        if metadata.get("content_type"):
            parts.append(f"Type: {metadata['content_type']}")
        
        # Add the main content
        parts.append(f"Content:\n{content.strip()}")
        
        # Add structural metadata
        if metadata.get("heading_level"):
            parts.append(f"Level: H{metadata['heading_level']}")
        
        return "\n".join(parts)
    
    def get_strategy_name(self) -> str:
        return "structured"


class CustomStrategy(EmbeddingStrategy):
    """Base class for user-defined custom strategies.
    
    Users can extend this class to implement their own
    text preparation logic.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration including custom function."""
        super().__init__(config)
        self.prepare_func = self.config.get("prepare_func")
        if not self.prepare_func:
            raise ValueError("CustomStrategy requires 'prepare_func' in config")
    
    def prepare_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """Use custom function to prepare text."""
        return self.prepare_func(content, metadata)
    
    def get_strategy_name(self) -> str:
        return self.config.get("name", "custom")


# Strategy factory
_STRATEGIES = {
    "simple": SimpleStrategy,
    "multi_vector": MultiVectorStrategy,
    "hybrid": HybridStrategy,
    "structured": StructuredStrategy,
    "custom": CustomStrategy
}


def create_strategy(
    strategy_name: str,
    config: Optional[Dict[str, Any]] = None
) -> EmbeddingStrategy:
    """Create an embedding strategy by name.
    
    Args:
        strategy_name: Name of the strategy
        config: Optional configuration for the strategy
        
    Returns:
        Configured embedding strategy
        
    Raises:
        ValueError: If strategy name is not recognized
    """
    if strategy_name not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available strategies: {list(_STRATEGIES.keys())}"
        )
    
    strategy_class = _STRATEGIES[strategy_name]
    return strategy_class(config)


def register_strategy(name: str, strategy_class: type):
    """Register a custom strategy class.
    
    This allows users to add their own strategies without
    modifying the library code.
    
    Args:
        name: Name to register the strategy under
        strategy_class: Strategy class (must inherit from EmbeddingStrategy)
    """
    if not issubclass(strategy_class, EmbeddingStrategy):
        raise ValueError("Strategy class must inherit from EmbeddingStrategy")
    
    _STRATEGIES[name] = strategy_class