"""Metadata enhancement using LLM for intelligent chunk augmentation."""

import asyncio
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field
from knowledge_core_engine.core.chunking.base import ChunkResult
from knowledge_core_engine.utils.config import get_settings
from knowledge_core_engine.utils.logger import get_logger, log_detailed, log_step
from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.generation.providers import create_llm_provider, LLMProvider

logger = get_logger(__name__)


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
        self.llm_provider: Optional[LLMProvider] = None
        self._initialized = False
        
        logger.info(f"MetadataEnhancer created with {self.config.llm_provider}")
    
    async def _ensure_initialized(self):
        """Ensure the enhancer is initialized."""
        if not self._initialized:
            await self._init_llm_provider()
            self._initialized = True
    
    async def _init_llm_provider(self):
        """Initialize LLM provider based on provider configuration."""
        if self.config.llm_provider == "mock":
            # Mock provider for testing
            self.llm_provider = None
            return
        
        # Get API key from config or environment
        if not self.config.api_key:
            settings = get_settings()
            if self.config.llm_provider == "deepseek":
                self.config.api_key = settings.deepseek_api_key
            elif self.config.llm_provider == "qwen":
                self.config.api_key = settings.qwen_api_key
        
        # Create RAG configuration for the provider
        rag_config = RAGConfig(
            llm_provider=self.config.llm_provider,
            llm_model=self.config.model_name,
            llm_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            extra_params={
                "max_retries": self.config.max_retries,
                "retry_delay": self.config.retry_delay
            }
        )
        
        try:
            # Create LLM provider
            self.llm_provider = await create_llm_provider(rag_config)
            logger.info(f"Created {self.config.llm_provider} LLM provider")
        except Exception as e:
            logger.error(f"Failed to create LLM provider: {e}")
            # Fallback to None to use placeholder responses
            self.llm_provider = None
    
    async def enhance_chunk(self, chunk: ChunkResult) -> ChunkResult:
        """Enhance a single chunk with LLM-generated metadata.
        
        Args:
            chunk: The chunk to enhance
            
        Returns:
            Enhanced chunk with additional metadata
        """
        await self._ensure_initialized()
        chunk_id = chunk.metadata.get('chunk_id', f'chunk_{id(chunk)}')
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(chunk)
            if self._cache and cache_key in self._cache:
                log_detailed(f"Cache hit for chunk enhancement", 
                           data={"chunk_id": chunk_id})
                cached_metadata = self._cache[cache_key]
                chunk.metadata.update(cached_metadata)
                return chunk
            
            log_detailed(f"Enhancing chunk", 
                       data={"chunk_id": chunk_id, 
                             "content_length": len(chunk.content)})
            
            # Build prompt
            prompt = self._build_enhancement_prompt(chunk.content)
            
            # Call LLM with retry
            response = await self._call_llm_with_retry(prompt)
            
            # Parse response
            metadata = await self._parse_llm_response(response)
            
            # Update chunk metadata
            enhanced_metadata = metadata.model_dump()
            chunk.metadata.update(enhanced_metadata)
            
            log_detailed(f"Enhancement successful", 
                       data={
                           "chunk_id": chunk_id,
                           "summary": enhanced_metadata.get('summary', '')[:50] + "...",
                           "num_questions": len(enhanced_metadata.get('questions', [])),
                           "chunk_type": enhanced_metadata.get('chunk_type', 'unknown')
                       })
            
            # Cache the result
            if self._cache is not None:
                self._cache[cache_key] = enhanced_metadata
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to enhance chunk {chunk_id}: {e}")
            # Mark as failed but return original chunk
            chunk.metadata["enhancement_failed"] = True
            chunk.metadata["enhancement_error"] = str(e)
            return chunk
    
    @log_step("Batch Metadata Enhancement")
    async def enhance_batch(self, chunks: List[ChunkResult]) -> List[ChunkResult]:
        """Enhance multiple chunks in batch.
        
        Args:
            chunks: List of chunks to enhance
            
        Returns:
            List of enhanced chunks
        """
        await self._ensure_initialized()
        log_detailed(f"Starting batch enhancement", 
                    data={"num_chunks": len(chunks), 
                          "max_concurrent": self.config.max_concurrent_requests})
        
        # Create tasks for concurrent processing
        tasks = []
        for chunk in chunks:
            task = self._enhance_with_semaphore(chunk)
            tasks.append(task)
        
        # Process all chunks
        enhanced_chunks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        result = []
        failed_count = 0
        for i, enhanced in enumerate(enhanced_chunks):
            if isinstance(enhanced, Exception):
                logger.error(f"Failed to enhance chunk {i}: {enhanced}")
                # Return original chunk with error flag
                chunks[i].metadata["enhancement_failed"] = True
                chunks[i].metadata["enhancement_error"] = str(enhanced)
                result.append(chunks[i])
                failed_count += 1
            else:
                result.append(enhanced)
        
        successful = len(chunks) - failed_count
        logger.info(f"Enhanced {successful}/{len(chunks)} chunks successfully")
        
        # Calculate cache hits
        cache_hits = 0
        if self._cache is not None:
            for c in result:
                if "enhancement_failed" not in c.metadata:
                    cache_key = self._get_cache_key(c)
                    if cache_key in self._cache:
                        cache_hits += 1
        
        log_detailed(f"Batch enhancement completed", 
                    data={"successful": successful, 
                          "failed": failed_count,
                          "cache_hits": cache_hits})
        
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
        
        # Use LLM provider if available
        if self.llm_provider:
            try:
                log_detailed("Calling LLM provider", 
                           data={"provider": self.config.llm_provider, "model": self.config.model_name})
                
                # Build messages for chat completion
                messages = [
                    {"role": "system", "content": "你是一个专业的文档分析助手，擅长提取文档的关键信息。请按照指定的JSON格式返回结果。"},
                    {"role": "user", "content": prompt}
                ]
                
                # Call provider
                response = await self.llm_provider.generate(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                # Extract content from response
                content = response.get("content", "")
                
                # Try to parse JSON from response
                try:
                    # First try direct parsing
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code block
                    import re
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(1))
                    else:
                        # Try to find any JSON object
                        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group(0))
                        else:
                            raise ValueError("No valid JSON found in response")
                
                # Validate response has required fields
                if all(key in result for key in ["summary", "questions", "chunk_type", "keywords"]):
                    return json.dumps(result)
                else:
                    logger.warning("LLM response missing required fields, using placeholder")
                    
            except Exception as e:
                logger.error(f"LLM provider call failed: {e}")
        
        # Fallback: return placeholder response
        logger.warning(f"Using placeholder response for {self.config.llm_provider}")
        return json.dumps({
            "summary": "文档内容摘要",
            "questions": ["这是关于什么的？", "如何使用？", "有什么优势？"],
            "chunk_type": "概念定义",
            "keywords": ["文档", "内容", "示例"]
        })
    
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
    
    async def close(self):
        """Clean up resources."""
        if self.llm_provider and hasattr(self.llm_provider, 'close'):
            await self.llm_provider.close()
            logger.info("LLM provider closed")