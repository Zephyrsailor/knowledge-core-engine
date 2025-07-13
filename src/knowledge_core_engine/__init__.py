"""
KnowledgeCore Engine - Next-generation knowledge base engine with advanced RAG capabilities.
"""

__version__ = "0.1.0"
__author__ = "Knowledge Core Team"

from knowledge_core_engine.utils.logger import setup_logger
from knowledge_core_engine.core.config import RAGConfig

# Initialize logger
logger = setup_logger(__name__)

# For simple_usage.py - these would need to be implemented
# from knowledge_core_engine.simple_api import RAG, Config, Document

__all__ = ["__version__", "__author__", "logger", "RAGConfig"]