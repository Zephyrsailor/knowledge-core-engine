"""Base provider interfaces and factory pattern.

All providers (LLM, Embedding, VectorDB) follow the same pattern:
1. Configuration through dict/yaml/env
2. Factory pattern for instantiation
3. Async interface for operations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass
import os
from enum import Enum


@dataclass
class ProviderConfig:
    """Base configuration for all providers."""
    provider: str  # e.g., "openai", "deepseek", "dashscope"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}
        
        # Try to load API key from environment if not provided
        if not self.api_key:
            env_key = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_key)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ProviderConfig":
        """Create config from dictionary."""
        return cls(**config)


class Provider(ABC):
    """Base interface for all providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @abstractmethod
    async def initialize(self):
        """Initialize the provider (e.g., test connection)."""
        pass
    
    @abstractmethod
    async def close(self):
        """Clean up resources."""
        pass


class ProviderFactory:
    """Factory for creating providers based on configuration."""
    
    _registry: Dict[str, Dict[str, Type[Provider]]] = {
        "llm": {},
        "embedding": {},
        "vectordb": {}
    }
    
    @classmethod
    def register(cls, provider_type: str, provider_name: str, provider_class: Type[Provider]):
        """Register a provider implementation.
        
        Args:
            provider_type: "llm", "embedding", or "vectordb"
            provider_name: Name of the provider (e.g., "openai", "deepseek")
            provider_class: The provider class to register
        """
        if provider_type not in cls._registry:
            raise ValueError(f"Invalid provider type: {provider_type}")
        
        cls._registry[provider_type][provider_name] = provider_class
    
    @classmethod
    def create(cls, provider_type: str, config: Dict[str, Any]) -> Provider:
        """Create a provider instance from configuration.
        
        Args:
            provider_type: "llm", "embedding", or "vectordb"
            config: Provider configuration dict
            
        Returns:
            Configured provider instance
        """
        if provider_type not in cls._registry:
            raise ValueError(f"Invalid provider type: {provider_type}")
        
        provider_config = ProviderConfig.from_dict(config)
        provider_name = provider_config.provider
        
        if provider_name not in cls._registry[provider_type]:
            available = list(cls._registry[provider_type].keys())
            raise ValueError(
                f"Unknown {provider_type} provider: {provider_name}. "
                f"Available: {available}"
            )
        
        provider_class = cls._registry[provider_type][provider_name]
        return provider_class(provider_config)
    
    @classmethod
    def list_providers(cls, provider_type: str) -> list:
        """List available providers of a given type."""
        if provider_type not in cls._registry:
            raise ValueError(f"Invalid provider type: {provider_type}")
        return list(cls._registry[provider_type].keys())