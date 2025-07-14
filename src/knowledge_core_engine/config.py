"""[DEPRECATED] Simple configuration for KnowledgeCore Engine.

WARNING: This file is deprecated and will be removed in a future version.
Please use:
- knowledge_core_engine.core.config.RAGConfig for RAG-specific configuration
- knowledge_core_engine.utils.config.get_settings() for environment settings

Just one way to do things - through environment variables and code.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Config:
    """All configuration in one place."""
    
    # LLM
    llm_provider: str = "deepseek"  # deepseek, qwen, openai
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_temperature: float = 0.1
    
    # Embedding  
    embedding_provider: str = "dashscope"  # dashscope, openai, huggingface
    embedding_api_key: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    
    # Vector DB
    vectordb_provider: str = "chromadb"  # chromadb, pinecone
    vectordb_persist_dir: str = "./data/chroma_db"
    vectordb_collection: str = "knowledge_core"
    
    # Strategy
    use_multi_vector: bool = True  # Combine content + summary + questions
    
    def __post_init__(self):
        """Load from environment if not provided."""
        # LLM
        if not self.llm_api_key:
            self.llm_api_key = os.getenv(f"{self.llm_provider.upper()}_API_KEY")
        
        # Embedding
        if not self.embedding_api_key:
            if self.embedding_provider == "dashscope":
                self.embedding_api_key = os.getenv("DASHSCOPE_API_KEY")
            else:
                self.embedding_api_key = os.getenv(f"{self.embedding_provider.upper()}_API_KEY")
        
        # Set default models
        if not self.llm_model:
            self.llm_model = {
                "deepseek": "deepseek-chat",
                "qwen": "qwen2.5-72b-instruct", 
                "openai": "gpt-4-turbo-preview"
            }.get(self.llm_provider)
        
        if not self.embedding_model:
            self.embedding_model = {
                "dashscope": "text-embedding-v3",
                "openai": "text-embedding-3-large",
                "huggingface": "BAAI/bge-large-zh-v1.5"
            }.get(self.embedding_provider)
        
        if not self.embedding_dim:
            self.embedding_dim = {
                "dashscope": 1536,
                "openai": 3072,
                "huggingface": 1024
            }.get(self.embedding_provider, 1536)