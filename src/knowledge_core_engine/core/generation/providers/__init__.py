"""LLM providers for generation."""

from typing import Dict, Any, List, AsyncGenerator, Optional
from abc import ABC, abstractmethod
import logging

from ...config import RAGConfig

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: RAGConfig):
        """Initialize provider.
        
        Args:
            config: RAG configuration
        """
        self.config = config
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Provider-specific parameters
            
        Returns:
            Response dict with 'content' and 'usage'
        """
        pass
    
    @abstractmethod
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream generate response.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature  
            max_tokens: Maximum tokens
            **kwargs: Provider-specific parameters
            
        Yields:
            Response chunks
        """
        pass


async def create_llm_provider(config: RAGConfig) -> LLMProvider:
    """Create LLM provider based on config.
    
    Args:
        config: RAG configuration
        
    Returns:
        LLM provider instance
    """
    provider_name = config.llm_provider.lower()
    
    if provider_name == "deepseek":
        from .deepseek import DeepSeekProvider
        return DeepSeekProvider(config)
    
    elif provider_name == "qwen":
        from .qwen import QwenProvider
        return QwenProvider(config)
    
    elif provider_name == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(config)
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = [
    "LLMProvider",
    "create_llm_provider"
]