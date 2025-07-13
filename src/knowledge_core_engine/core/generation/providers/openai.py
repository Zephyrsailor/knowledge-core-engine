"""OpenAI LLM provider implementation (for reference/testing)."""

import os
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, AsyncGenerator, Optional
import logging

from . import LLMProvider
from ...config import RAGConfig

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    API_BASE = "https://api.openai.com/v1"
    
    def __init__(self, config: RAGConfig):
        """Initialize OpenAI provider.
        
        Args:
            config: RAG configuration
        """
        super().__init__(config)
        self.api_key = config.llm_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = config.llm_model or "gpt-4-turbo-preview"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response from OpenAI.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Response dict
        """
        url = f"{self.API_BASE}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Add extra parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Extract response
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    
                    return {
                        "content": content,
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                    }
                    
            except aiohttp.ClientError as e:
                logger.error(f"OpenAI API error: {e}")
                raise Exception(f"OpenAI API call failed: {e}")
    
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream generate response from OpenAI.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        url = f"{self.API_BASE}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    response.raise_for_status()
                    
                    # Process stream
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if not line or line == "data: [DONE]":
                            continue
                        
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                
                                # Extract delta
                                choice = data["choices"][0]
                                delta = choice.get("delta", {})
                                
                                if "content" in delta:
                                    yield {
                                        "content": delta["content"],
                                        "usage": None
                                    }
                                
                                # Check if finished
                                if choice.get("finish_reason"):
                                    yield {
                                        "content": "",
                                        "usage": None,  # OpenAI doesn't provide usage in stream
                                        "is_final": True
                                    }
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse SSE data: {line}")
                                continue
                    
            except aiohttp.ClientError as e:
                logger.error(f"OpenAI streaming error: {e}")
                raise Exception(f"OpenAI streaming failed: {e}")