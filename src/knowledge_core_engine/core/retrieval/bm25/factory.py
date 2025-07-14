"""Factory for creating BM25 retriever instances."""

import logging
from typing import Optional

from ...config import RAGConfig
from .base import BaseBM25Retriever
from .bm25s_retriever import BM25SRetriever
from .elasticsearch_retriever import ElasticsearchRetriever

logger = logging.getLogger(__name__)


def create_bm25_retriever(config: RAGConfig) -> Optional[BaseBM25Retriever]:
    """Create a BM25 retriever based on configuration.
    
    Args:
        config: RAG configuration object
        
    Returns:
        BM25 retriever instance or None if BM25 is not needed
    """
    # Check if BM25 is needed
    retrieval_strategy = getattr(config, "retrieval_strategy", "vector")
    if retrieval_strategy not in ["bm25", "hybrid"]:
        logger.info("BM25 not needed for retrieval strategy: %s", retrieval_strategy)
        return None
    
    # Get BM25 provider
    bm25_provider = getattr(config, "bm25_provider", "bm25s")
    
    if bm25_provider == "bm25s":
        return _create_bm25s_retriever(config)
    elif bm25_provider == "elasticsearch":
        return _create_elasticsearch_retriever(config)
    elif bm25_provider == "none":
        return None
    else:
        logger.warning(
            "Unknown BM25 provider: %s, falling back to bm25s", 
            bm25_provider
        )
        return _create_bm25s_retriever(config)


def _create_bm25s_retriever(config: RAGConfig) -> BM25SRetriever:
    """Create BM25S retriever."""
    # Get BM25 parameters
    k1 = getattr(config, "bm25_k1", 1.5)
    b = getattr(config, "bm25_b", 0.75)
    epsilon = getattr(config, "bm25_epsilon", 0.25)
    
    # Detect language from config or default
    language = getattr(config, "language", "en")
    if language == "zh" or language == "chinese":
        language = "zh"
    elif language in ["en", "english"]:
        language = "en"
    else:
        language = "multi"  # Multilingual
    
    logger.info(
        "Creating BM25S retriever (k1=%.2f, b=%.2f, language=%s)",
        k1, b, language
    )
    
    return BM25SRetriever(
        k1=k1,
        b=b,
        epsilon=epsilon,
        language=language
    )


def _create_elasticsearch_retriever(config: RAGConfig) -> ElasticsearchRetriever:
    """Create Elasticsearch retriever."""
    # Get Elasticsearch configuration
    es_url = getattr(config, "elasticsearch_url", None)
    if not es_url:
        raise ValueError(
            "Elasticsearch URL not provided. "
            "Please set 'elasticsearch_url' in configuration."
        )
    
    index_name = getattr(config, "elasticsearch_index", "knowledge_core")
    k1 = getattr(config, "bm25_k1", 1.2)
    b = getattr(config, "bm25_b", 0.75)
    
    # Additional ES options
    es_options = {}
    if hasattr(config, "elasticsearch_username"):
        es_options["basic_auth"] = (
            config.elasticsearch_username,
            getattr(config, "elasticsearch_password", "")
        )
    
    logger.info(
        "Creating Elasticsearch retriever (url=%s, index=%s)",
        es_url, index_name
    )
    
    return ElasticsearchRetriever(
        es_url=es_url,
        index_name=index_name,
        k1=k1,
        b=b,
        **es_options
    )