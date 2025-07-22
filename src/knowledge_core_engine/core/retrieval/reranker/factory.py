"""Factory for creating reranker instances."""

import logging
from typing import Optional, Dict, Any

from ...config import RAGConfig
from .base import BaseReranker
from .api_reranker import APIReranker

logger = logging.getLogger(__name__)


def create_reranker(config: RAGConfig) -> Optional[BaseReranker]:
    """Create a reranker instance based on configuration.
    
    Args:
        config: RAG configuration object
        
    Returns:
        Reranker instance or None if reranking is disabled
    """
    if not config.enable_reranking:
        logger.info("Reranking is disabled")
        return None
    
    provider = getattr(config, "reranker_provider", "huggingface")
    
    if provider == "huggingface":
        return _create_huggingface_reranker(config)
    elif provider == "api":
        return _create_api_reranker(config)
    elif provider == "none":
        return None
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")


def _create_huggingface_reranker(config: RAGConfig) -> "HuggingFaceReranker":
    """Create HuggingFace reranker.
    
    This function lazily imports HuggingFaceReranker to avoid torch dependency
    when not using HuggingFace reranker.
    """
    try:
        from .huggingface_reranker import HuggingFaceReranker
    except ImportError as e:
        if "torch" in str(e).lower():
            raise ImportError(
                "PyTorch is required for HuggingFace reranker. "
                "Please install it with: pip install 'knowledge-core-engine[reranker-hf]'"
            )
        raise
    
    model_name = config.reranker_model or "bge-reranker-v2-m3"
    use_fp16 = getattr(config, "use_fp16", True)
    device = getattr(config, "reranker_device", None)
    
    logger.info(f"Creating HuggingFace reranker with model: {model_name}")
    
    return HuggingFaceReranker(
        model_name=model_name,
        use_fp16=use_fp16,
        device=device
    )


def _create_api_reranker(config: RAGConfig) -> APIReranker:
    """Create API reranker."""
    # Determine provider from model name or explicit setting
    api_provider = getattr(config, "reranker_api_provider", None)
    
    if not api_provider:
        # Infer from model name
        model = config.reranker_model or ""
        if "gte-rerank" in model:
            api_provider = "dashscope"
        elif "cohere" in model:
            api_provider = "cohere"
        elif "jina" in model:
            api_provider = "jina"
        else:
            api_provider = "dashscope"  # Default
    
    api_key = getattr(config, f"{api_provider}_api_key", None)
    model = config.reranker_model
    timeout = getattr(config, "api_timeout", 30)
    
    logger.info(f"Creating API reranker with provider: {api_provider}")
    
    return APIReranker(
        provider=api_provider,
        api_key=api_key,
        model=model,
        timeout=timeout
    )