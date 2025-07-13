"""Qwen (Tongyi Qianwen) LLM provider implementation."""

import os
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, AsyncGenerator, Optional
import logging

from . import LLMProvider
from ...config import RAGConfig

logger = logging.getLogger(__name__)


class QwenProvider(LLMProvider):
    """Qwen/Tongyi LLM provider via DashScope."""
    
    API_BASE = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation"
    
    def __init__(self, config: RAGConfig):
        """Initialize Qwen provider.
        
        Args:
            config: RAG configuration
        """
        super().__init__(config)
        self.api_key = config.llm_api_key or os.getenv("DASHSCOPE_API_KEY")
        
        if not self.api_key:
            raise ValueError("DashScope API key is required for Qwen")
        
        self.model = config.llm_model or "qwen2.5-72b-instruct"
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
        """Generate response from Qwen.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Response dict
        """
        url = f"{self.API_BASE}/generation"
        
        # Convert messages format
        formatted_messages = self._format_messages(messages)
        
        payload = {
            "model": self.model,
            "input": {
                "messages": formatted_messages
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "enable_search": False,  # Disable web search
                "stream": False
            }
        }
        
        # Add extra parameters
        for key, value in kwargs.items():
            if key not in ["model", "input", "parameters"]:
                payload["parameters"][key] = value
        
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
                    
                    # Check for API errors
                    if data.get("code"):
                        raise Exception(f"Qwen API error: {data.get('message', 'Unknown error')}")
                    
                    # Extract response
                    output = data.get("output", {})
                    content = output.get("text", "")
                    usage = data.get("usage", {})
                    
                    return {
                        "content": content,
                        "usage": {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                    }
                    
            except aiohttp.ClientError as e:
                logger.error(f"Qwen API error: {e}")
                raise Exception(f"Qwen API call failed: {e}")
    
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream generate response from Qwen.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        url = f"{self.API_BASE}/generation"
        
        # Convert messages format
        formatted_messages = self._format_messages(messages)
        
        payload = {
            "model": self.model,
            "input": {
                "messages": formatted_messages
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "enable_search": False,
                "stream": True,
                "incremental_output": True  # Only output new content
            }
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
                    
                    # Process SSE stream
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if not line:
                            continue
                        
                        # Qwen uses different SSE format
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        
                        try:
                            data = json.loads(line)
                            
                            # Check for errors
                            if data.get("code"):
                                logger.error(f"Qwen stream error: {data}")
                                continue
                            
                            output = data.get("output", {})
                            
                            # Check if finished
                            if output.get("finish_reason"):
                                # Final chunk with usage
                                usage = data.get("usage", {})
                                yield {
                                    "content": "",
                                    "usage": {
                                        "prompt_tokens": usage.get("input_tokens", 0),
                                        "completion_tokens": usage.get("output_tokens", 0),
                                        "total_tokens": usage.get("total_tokens", 0)
                                    },
                                    "is_final": True
                                }
                            else:
                                # Content chunk
                                content = output.get("text", "")
                                if content:
                                    yield {
                                        "content": content,
                                        "usage": None
                                    }
                                    
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE data: {line}")
                            continue
                    
            except aiohttp.ClientError as e:
                logger.error(f"Qwen streaming error: {e}")
                raise Exception(f"Qwen streaming failed: {e}")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Qwen API.
        
        Args:
            messages: Standard chat messages
            
        Returns:
            Formatted messages
        """
        # Qwen uses the same format but might need adjustments
        formatted = []
        
        for msg in messages:
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        return formatted
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        # Use DashScope tokenizer endpoint if available
        # For now, use approximation
        import re
        
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        
        # Qwen tokenizer is similar to GPT
        return chinese_chars // 2 + other_chars // 4