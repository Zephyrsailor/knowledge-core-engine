"""Metadata enhancement using LLM for intelligent chunk augmentation."""

import asyncio
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
from datetime import datetime

from pydantic import BaseModel, Field
from knowledge_core_engine.core.chunking.base import ChunkResult
from knowledge_core_engine.utils.config import get_settings

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    """LLM-generated metadata structure."""
    summary: str = Field(..., description="一句话摘要")
    questions: List[str] = Field(..., description="3-5个潜在问题")
    chunk_type: str = Field(..., description="分类标签")
    keywords: List[str] = Field(..., description="关键词提取")


@dataclass
class EnhancementConfig:
    """Configuration for metadata enhancement."""
    llm_provider: str = "deepseek"
    model_name: str = "deepseek-chat"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 500
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_cache: bool = True
    cache_ttl: int = 86400  # 24 hours
    max_concurrent_requests: int = 10
    
    # Chunk type options
    chunk_type_options: List[str] = field(default_factory=lambda: [
        "概念定义", "操作步骤", "示例代码", 
        "理论说明", "问题解答", "其他"
    ])
    
    # Prompt template
    prompt_template: str = """分析以下文本内容，生成结构化元数据：

文本内容：
{content}

请生成以下信息：
1. summary: 用一句话概括内容要点（20-50字）
2. questions: 列出3-5个用户可能会问的问题
3. chunk_type: 从以下选项中选择一个类型标签：[{chunk_types}]
4. keywords: 提取3-8个关键词

要求：
- summary要简洁准确，抓住核心要点
- questions要符合用户实际需求，避免过于宽泛
- keywords要包含专业术语和核心概念
- 严格按照JSON格式返回，确保可以被解析

JSON格式示例：
{{
    "summary": "RAG技术通过结合检索和生成提升AI回答质量",
    "questions": ["什么是RAG技术？", "RAG如何工作？", "RAG有哪些应用场景？"],
    "chunk_type": "概念定义",
    "keywords": ["RAG", "检索增强生成", "AI", "语言模型"]
}}"""


class MetadataEnhancer:
    """Enhance chunk metadata using LLM."""
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        """Initialize the metadata enhancer.
        
        Args:
            config: Enhancement configuration
        """
        self.config = config or EnhancementConfig()
        self._cache = {} if self.config.enable_cache else None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Initialize LLM client based on provider
        self._init_llm_client()
        
        logger.info(f"MetadataEnhancer initialized with {self.config.llm_provider}")
    
    def _init_llm_client(self):
        """Initialize LLM client based on provider."""
        if self.config.llm_provider == "mock":
            # Mock client for testing
            self.llm_client = None
            return
        
        # Get API key from config or environment
        if not self.config.api_key:
            settings = get_settings()
            if self.config.llm_provider == "deepseek":
                self.config.api_key = settings.deepseek_api_key
            elif self.config.llm_provider == "qwen":
                self.config.api_key = settings.qwen_api_key
        
        # TODO: Initialize actual LLM client (OpenAI-compatible API)
        # For now, we'll implement a placeholder
        self.llm_client = None
    
    async def enhance_chunk(self, chunk: ChunkResult) -> ChunkResult:
        """Enhance a single chunk with LLM-generated metadata.
        
        Args:
            chunk: The chunk to enhance
            
        Returns:
            Enhanced chunk with additional metadata
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(chunk)
            if self._cache and cache_key in self._cache:
                logger.debug(f"Cache hit for chunk {chunk.metadata.get('chunk_id', 'unknown')}")
                cached_metadata = self._cache[cache_key]
                chunk.metadata.update(cached_metadata)
                return chunk
            
            # Build prompt
            prompt = self._build_enhancement_prompt(chunk.content)
            
            # Call LLM with retry
            response = await self._call_llm_with_retry(prompt)
            
            # Parse response
            metadata = await self._parse_llm_response(response)
            
            # Update chunk metadata
            enhanced_metadata = metadata.model_dump()
            chunk.metadata.update(enhanced_metadata)
            
            # Cache the result
            if self._cache is not None:
                self._cache[cache_key] = enhanced_metadata
            
            logger.debug(f"Enhanced chunk {chunk.metadata.get('chunk_id', 'unknown')}")
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to enhance chunk: {e}")
            # Mark as failed but return original chunk
            chunk.metadata["enhancement_failed"] = True
            chunk.metadata["enhancement_error"] = str(e)
            return chunk
    
    async def enhance_batch(self, chunks: List[ChunkResult]) -> List[ChunkResult]:
        """Enhance multiple chunks in batch.
        
        Args:
            chunks: List of chunks to enhance
            
        Returns:
            List of enhanced chunks
        """
        logger.info(f"Enhancing batch of {len(chunks)} chunks")
        
        # Create tasks for concurrent processing
        tasks = []
        for chunk in chunks:
            task = self._enhance_with_semaphore(chunk)
            tasks.append(task)
        
        # Process all chunks
        enhanced_chunks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        result = []
        for i, enhanced in enumerate(enhanced_chunks):
            if isinstance(enhanced, Exception):
                logger.error(f"Failed to enhance chunk {i}: {enhanced}")
                # Return original chunk with error flag
                chunks[i].metadata["enhancement_failed"] = True
                chunks[i].metadata["enhancement_error"] = str(enhanced)
                result.append(chunks[i])
            else:
                result.append(enhanced)
        
        successful = sum(1 for c in result if not c.metadata.get("enhancement_failed"))
        logger.info(f"Enhanced {successful}/{len(chunks)} chunks successfully")
        
        return result
    
    async def _enhance_with_semaphore(self, chunk: ChunkResult) -> ChunkResult:
        """Enhance chunk with rate limiting."""
        async with self._semaphore:
            return await self.enhance_chunk(chunk)
    
    def _build_enhancement_prompt(self, content: str) -> str:
        """Build the enhancement prompt for LLM.
        
        Args:
            content: The chunk content
            
        Returns:
            Formatted prompt
        """
        chunk_types = ", ".join(self.config.chunk_type_options)
        return self.config.prompt_template.format(
            content=content,
            chunk_types=chunk_types
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response
        """
        if self.config.llm_provider == "mock":
            # Mock response for testing
            return json.dumps({
                "summary": "This is a mock summary of the content",
                "questions": ["What is this about?", "How does it work?", "What are the benefits?"],
                "chunk_type": "概念定义",
                "keywords": ["test", "mock", "example"]
            })
        
        # TODO: Implement actual LLM API calls
        # For now, raise NotImplementedError
        raise NotImplementedError(f"LLM provider {self.config.llm_provider} not implemented yet")
    
    async def _call_llm_with_retry(self, prompt: str) -> str:
        """Call LLM with retry logic.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await self._call_llm(prompt)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"LLM call failed after {self.config.max_retries} attempts: {e}")
        
        raise last_error
    
    async def _parse_llm_response(self, response: str) -> ChunkMetadata:
        """Parse LLM response into structured metadata.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed ChunkMetadata
        """
        try:
            # Parse JSON response
            data = json.loads(response)
            
            # Validate and create ChunkMetadata
            metadata = ChunkMetadata(
                summary=data["summary"],
                questions=data["questions"],
                chunk_type=data["chunk_type"],
                keywords=data["keywords"]
            )
            
            # Additional validation
            if metadata.chunk_type not in self.config.chunk_type_options:
                logger.warning(f"Invalid chunk_type '{metadata.chunk_type}', defaulting to '其他'")
                metadata.chunk_type = "其他"
            
            # Limit questions to 5
            if len(metadata.questions) > 5:
                metadata.questions = metadata.questions[:5]
            
            # Limit keywords to 8
            if len(metadata.keywords) > 8:
                metadata.keywords = metadata.keywords[:8]
            
            return metadata
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response}")
            raise ValueError(f"Invalid LLM response format: {e}")
    
    def _get_cache_key(self, chunk: ChunkResult) -> str:
        """Generate cache key for a chunk.
        
        Args:
            chunk: The chunk to generate key for
            
        Returns:
            Cache key
        """
        # Use content hash as cache key
        content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
        return f"enhance_{content_hash}"
    
    def clear_cache(self):
        """Clear the enhancement cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Enhancement cache cleared")