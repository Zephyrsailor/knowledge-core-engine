"""LLM provider abstraction for flexible LLM integration."""

from abc import abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass

from .base import Provider, ProviderConfig, ProviderFactory


@dataclass
class LLMConfig(ProviderConfig):
    """LLM-specific configuration."""
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False


@dataclass
class LLMResponse:
    """Standard LLM response format."""
    content: str
    model: str
    usage: Dict[str, int]  # tokens, etc.
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMProvider(Provider):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion for a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse with generated text
        """
        pass
    
    @abstractmethod
    async def complete_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream completion for a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Yields:
            Generated text chunks
        """
        pass
    
    async def complete_json(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate JSON output following a schema.
        
        Args:
            prompt: Input prompt
            schema: Expected JSON schema
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON response
        """
        # Default implementation - providers can override for better support
        import json
        
        json_prompt = f"{prompt}\n\nReturn your response as valid JSON following this schema:\n{json.dumps(schema, indent=2)}"
        response = await self.complete(json_prompt, **kwargs)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Failed to parse JSON from response: {response.content}")


# --- DeepSeek Implementation ---

class DeepSeekProvider(LLMProvider):
    """DeepSeek LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.deepseek.com/v1"
        self.model = config.model or "deepseek-chat"
    
    async def initialize(self):
        """Initialize DeepSeek client."""
        # Test API key validity
        if not self.config.api_key:
            raise ValueError("DeepSeek API key is required")
    
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using DeepSeek API."""
        # Implementation would use aiohttp or httpx
        # This is a placeholder
        return LLMResponse(
            content="DeepSeek response placeholder",
            model=self.model,
            usage={"total_tokens": 100}
        )
    
    async def complete_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream completion using DeepSeek API."""
        # Placeholder implementation
        yield "DeepSeek"
        yield " streaming"
        yield " response"
    
    async def close(self):
        """Clean up resources."""
        pass


# --- Qwen Implementation ---

class QwenProvider(LLMProvider):
    """Qwen (通义千问) LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://dashscope.aliyuncs.com/api/v1"
        self.model = config.model or "qwen2.5-72b-instruct"
    
    async def initialize(self):
        """Initialize Qwen client."""
        if not self.config.api_key:
            raise ValueError("DashScope API key is required for Qwen")
    
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Qwen API."""
        # Placeholder
        return LLMResponse(
            content="Qwen response placeholder",
            model=self.model,
            usage={"total_tokens": 100}
        )
    
    async def complete_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream completion using Qwen API."""
        yield "Qwen streaming response"
    
    async def close(self):
        """Clean up resources."""
        pass


# --- OpenAI Compatible Implementation ---

class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider (works with OpenAI, Azure, etc.)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.openai.com/v1"
        self.model = config.model or "gpt-4-turbo-preview"
    
    async def initialize(self):
        """Initialize OpenAI client."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
    
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using OpenAI API."""
        # Placeholder
        return LLMResponse(
            content="OpenAI response placeholder",
            model=self.model,
            usage={"total_tokens": 100}
        )
    
    async def complete_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream completion using OpenAI API."""
        yield "OpenAI streaming response"
    
    async def close(self):
        """Clean up resources."""
        pass


# Register providers
ProviderFactory.register("llm", "deepseek", DeepSeekProvider)
ProviderFactory.register("llm", "qwen", QwenProvider)
ProviderFactory.register("llm", "openai", OpenAIProvider)


# --- Usage Example ---
"""
# Configuration from YAML/JSON/Dict
config = {
    "provider": "deepseek",
    "api_key": "sk-xxx",  # or from env: DEEPSEEK_API_KEY
    "model": "deepseek-chat",
    "temperature": 0.1,
    "max_tokens": 2048
}

# Create LLM provider
llm = ProviderFactory.create("llm", config)
await llm.initialize()

# Use it
response = await llm.complete("What is RAG?")
print(response.content)

# Stream response
async for chunk in llm.complete_stream("Explain RAG in detail"):
    print(chunk, end="")

# Generate structured output
schema = {
    "summary": "string",
    "keywords": ["string"],
    "questions": ["string"]
}
result = await llm.complete_json("Analyze this text...", schema)
"""