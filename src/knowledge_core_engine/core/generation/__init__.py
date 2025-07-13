"""Generation module for RAG system."""

from .generator import Generator, GenerationResult, CitationReference
from .prompt_builder import PromptBuilder, PromptTemplate, ContextFormatter
from .citation_manager import CitationManager, Citation, CitationStyle

__all__ = [
    "Generator",
    "GenerationResult", 
    "CitationReference",
    "PromptBuilder",
    "PromptTemplate",
    "ContextFormatter",
    "CitationManager",
    "Citation",
    "CitationStyle"
]