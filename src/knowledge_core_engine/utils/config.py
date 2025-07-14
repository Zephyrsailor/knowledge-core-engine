"""Configuration management for KnowledgeCore Engine.

This module handles environment-based configuration using Pydantic Settings.
It manages API keys, file paths, and application-level settings from .env files.

Usage:
    from knowledge_core_engine.utils.config import get_settings
    
    settings = get_settings()
    api_key = settings.dashscope_api_key

For RAG-specific configuration, see core.config.RAGConfig.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Keys
    llama_cloud_api_key: str
    dashscope_api_key: str
    deepseek_api_key: Optional[str] = None
    
    # Model Configuration
    embedding_model: str = "text-embedding-v3"
    llm_model: str = "deepseek-chat"
    reranker_model: str = "bge-reranker-v2-m3-qwen"
    
    # Database Configuration
    chroma_persist_directory: Path = Path("./data/chroma_db")
    chroma_collection_name: str = "knowledge_core"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = Path("./logs/knowledge_core.log")
    
    # Cache Configuration
    enable_cache: bool = True
    cache_dir: Path = Path("./data/cache")
    cache_ttl: int = 86400  # 24 hours
    
    # LlamaParse Configuration
    llama_parse_result_type: str = "markdown"
    llama_parse_verbose: bool = True
    
    # Chunking Configuration
    chunk_size: int = 1024
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    
    # Retrieval Configuration
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    hybrid_search_alpha: float = 0.5
    
    # Generation Configuration
    generation_temperature: float = 0.1
    generation_max_tokens: int = 2048
    generation_timeout: int = 30
    
    # Evaluation Configuration
    eval_batch_size: int = 10
    eval_llm_model: str = "deepseek-chat"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()