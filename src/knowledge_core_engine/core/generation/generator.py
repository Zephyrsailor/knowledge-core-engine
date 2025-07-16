"""Generator module for answer generation with citations."""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

from ..config import RAGConfig
from ..retrieval.retriever import RetrievalResult
from .prompt_builder import PromptBuilder
from .citation_manager import CitationManager, Citation, CitationStyle
from .providers import create_llm_provider, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class CitationReference:
    """Reference to a source document."""
    index: int
    chunk_id: str
    document_title: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "document_title": self.document_title,
            "page": self.page,
            "text": self.text
        }
    
    def format_inline(self) -> str:
        """Format as inline citation."""
        return f"[{self.index}]"
    
    def format_footnote(self) -> str:
        """Format as footnote."""
        parts = []
        if self.document_title:
            parts.append(self.document_title)
        if self.page:
            parts.append(f"p.{self.page}")
        return f"[{self.index}] " + ", ".join(parts) if parts else f"[{self.index}]"


@dataclass
class GenerationResult:
    """Result from generation."""
    query: str
    answer: str
    citations: List[CitationReference] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "usage": self.usage,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    def get_formatted_answer(self, citation_style: str = "inline") -> str:
        """Get answer with formatted citations."""
        if not self.citations:
            return self.answer
        
        formatted = self.answer
        
        if citation_style == "footnote":
            # Add footnotes at the end
            footnotes = "\n\n---\n参考文献：\n"
            for citation in self.citations:
                footnotes += f"{citation.format_footnote()}\n"
            formatted += footnotes
        
        return formatted


@dataclass
class StreamChunk:
    """A chunk in streaming generation."""
    content: str
    is_final: bool = False
    usage: Optional[Dict[str, Any]] = None
    citations: Optional[List[CitationReference]] = None


class Generator:
    """Main generator for creating answers with citations."""
    
    def __init__(self, config: RAGConfig):
        """Initialize generator.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self._llm_client: Optional[LLMProvider] = None
        self._prompt_builder = PromptBuilder(config)
        self._citation_manager = CitationManager()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the generator."""
        if self._initialized:
            return
        
        # Create LLM client
        self._llm_client = await self._create_llm_client()
        
        self._initialized = True
        logger.info(f"Generator initialized with {self.config.llm_provider}")
    
    async def _create_llm_client(self) -> LLMProvider:
        """Create LLM client based on config."""
        return await create_llm_provider(self.config)
    
    async def generate(
        self,
        query: str,
        contexts: List[RetrievalResult],
        **kwargs
    ) -> GenerationResult:
        """Generate answer based on query and contexts.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            **kwargs: Additional generation parameters
            
        Returns:
            Generation result with answer and citations
        """
        if not self._initialized:
            await self.initialize()
        
        # Handle empty contexts
        if not contexts:
            return await self._generate_no_context_response(query)
        
        # Truncate contexts if needed
        contexts = self._truncate_contexts(contexts)
        
        # Build prompt
        prompt = self._prompt_builder.build_prompt(
            query=query,
            contexts=contexts,
            include_citations=self.config.include_citations,
            **kwargs
        )
        
        # Generate response
        try:
            response = await self._call_llm(prompt)
            
            # Extract answer and citations
            answer = response.get("content", "")
            usage = response.get("usage", {})
            
            # Process citations if enabled
            citations = []
            if self.config.include_citations:
                citations = self._extract_and_map_citations(answer, contexts)
            
            return GenerationResult(
                query=query,
                answer=answer,
                citations=citations,
                usage=usage,
                metadata={
                    "model": self.config.llm_model,
                    "temperature": self.config.temperature,
                    "context_count": len(contexts)
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if self.config.extra_params.get("max_retries", 0) > 0:
                return await self._retry_generation(query, contexts, **kwargs)
            raise
    
    async def stream_generate(
        self,
        query: str,
        contexts: List[RetrievalResult],
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream generate answer.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            **kwargs: Additional parameters
            
        Yields:
            Stream chunks
        """
        if not self._initialized:
            await self.initialize()
        
        # Build prompt
        prompt = self._prompt_builder.build_prompt(
            query=query,
            contexts=contexts,
            include_citations=self.config.include_citations,
            **kwargs
        )
        
        # Stream from LLM
        accumulated_text = ""
        
        async for chunk in self._stream_llm(prompt):
            content = chunk.get("content", "")
            accumulated_text += content
            
            if chunk.get("usage"):
                # Final chunk
                citations = []
                if self.config.include_citations:
                    citations = self._extract_and_map_citations(
                        accumulated_text, contexts
                    )
                
                yield StreamChunk(
                    content=content,
                    is_final=True,
                    usage=chunk["usage"],
                    citations=citations
                )
            else:
                yield StreamChunk(content=content)
    
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM with prompt.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response
        """
        messages = self._prompt_builder.build_messages(
            prompt=prompt,
            system_prompt=self.config.extra_params.get("system_prompt")
        )
        
        response = await self._llm_client.generate(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response
    
    async def _stream_llm(self, prompt: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream from LLM.
        
        Args:
            prompt: The prompt
            
        Yields:
            Response chunks
        """
        messages = self._prompt_builder.build_messages(
            prompt=prompt,
            system_prompt=self.config.extra_params.get("system_prompt")
        )
        
        async for chunk in self._llm_client.stream_generate(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        ):
            yield chunk
    
    def _truncate_contexts(
        self,
        contexts: List[RetrievalResult],
        max_tokens: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Truncate contexts to fit token limit.
        
        Args:
            contexts: Original contexts
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated contexts
        """
        if not max_tokens:
            max_tokens = self.config.extra_params.get(
                "max_context_tokens",
                self.config.max_tokens * 0.7  # Reserve 30% for response
            )
        
        # Simple truncation - keep top scoring contexts
        truncated = []
        total_tokens = 0
        
        for context in contexts:
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            estimated_tokens = len(context.content) // 4
            
            if total_tokens + estimated_tokens > max_tokens:
                break
            
            truncated.append(context)
            total_tokens += estimated_tokens
        
        return truncated
    
    def _extract_citations(self, text: str) -> List[int]:
        """Extract citation indices from text.
        
        Args:
            text: Text with citations like [1], [2]
            
        Returns:
            List of citation indices
        """
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]
    
    def _extract_and_map_citations(
        self,
        answer: str,
        contexts: List[RetrievalResult]
    ) -> List[CitationReference]:
        """Extract citations from answer and map to contexts.
        
        Args:
            answer: Generated answer with citations
            contexts: Source contexts
            
        Returns:
            List of citation references
        """
        # Extract citation indices
        indices = self._extract_citations(answer)
        
        # Map to contexts
        citations = []
        seen_indices = set()
        
        for idx in indices:
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            
            # Map index to context (1-based indexing)
            if 1 <= idx <= len(contexts):
                context = contexts[idx - 1]
                # 获取文档标题
                doc_title = context.metadata.get("document_title") or context.metadata.get("source") or context.metadata.get("file_path", "Unknown")
                
                # 如果有层级信息，添加章节路径
                if 'hierarchy' in context.metadata:
                    hierarchy = context.metadata['hierarchy']
                    if isinstance(hierarchy, str):
                        import json
                        hierarchy = json.loads(hierarchy)
                    
                    # 获取层级路径
                    if 'hierarchy_path' in hierarchy and len(hierarchy['hierarchy_path']) > 1:
                        # 跳过第一级（通常是文档标题）
                        section_path = ' > '.join(hierarchy['hierarchy_path'][1:])
                        doc_title = f"{doc_title} - {section_path}"
                
                citation = CitationReference(
                    index=idx,
                    chunk_id=context.chunk_id,
                    document_title=doc_title,
                    page=context.metadata.get("page"),
                    text=context.content[:200] + "..." if len(context.content) > 200 else context.content
                )
                citations.append(citation)
        
        return citations
    
    async def _generate_no_context_response(self, query: str) -> GenerationResult:
        """Generate response when no contexts are found.
        
        Args:
            query: User query
            
        Returns:
            Generation result
        """
        prompt = self._prompt_builder.build_no_context_prompt(query)
        
        response = await self._call_llm(prompt)
        
        return GenerationResult(
            query=query,
            answer=response.get("content", "抱歉，我在提供的文档中没有找到相关信息来回答您的问题。"),
            citations=[],
            usage=response.get("usage", {}),
            metadata={"no_context": True}
        )
    
    async def _retry_generation(
        self,
        query: str,
        contexts: List[RetrievalResult],
        retry_count: int = 0,
        **kwargs
    ) -> GenerationResult:
        """Retry generation with backoff.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            retry_count: Current retry count
            **kwargs: Additional parameters
            
        Returns:
            Generation result
        """
        max_retries = self.config.extra_params.get("max_retries", 3)
        
        if retry_count >= max_retries:
            raise Exception("Max retries exceeded")
        
        # Exponential backoff
        await asyncio.sleep(2 ** retry_count)
        
        # Reduce temperature on retry
        original_temp = self.config.temperature
        self.config.temperature *= (1 - self.config.extra_params.get("temperature_decay", 0.1))
        
        try:
            result = await self.generate(query, contexts, **kwargs)
            return result
        except Exception as e:
            logger.warning(f"Retry {retry_count + 1} failed: {e}")
            return await self._retry_generation(
                query, contexts, retry_count + 1, **kwargs
            )
        finally:
            self.config.temperature = original_temp
    
    async def _generate_with_critique(
        self,
        query: str,
        contexts: List[RetrievalResult],
        **kwargs
    ) -> GenerationResult:
        """Generate with self-critique.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            **kwargs: Additional parameters
            
        Returns:
            Improved generation result
        """
        # First generation
        initial_result = await self.generate(query, contexts, **kwargs)
        
        # Critique prompt
        critique_prompt = self._prompt_builder.build_critique_prompt(
            query=query,
            answer=initial_result.answer,
            contexts=contexts
        )
        
        # Get critique
        critique_response = await self._call_llm(critique_prompt)
        
        # Improve based on critique
        improvement_prompt = self._prompt_builder.build_improvement_prompt(
            query=query,
            answer=initial_result.answer,
            critique=critique_response.get("content", ""),
            contexts=contexts
        )
        
        final_response = await self._call_llm(improvement_prompt)
        
        # Extract citations from improved answer
        citations = []
        if self.config.include_citations:
            citations = self._extract_and_map_citations(
                final_response.get("content", ""),
                contexts
            )
        
        return GenerationResult(
            query=query,
            answer=final_response.get("content", ""),
            citations=citations,
            usage={
                "total_tokens": (
                    initial_result.usage.get("total_tokens", 0) +
                    critique_response.get("usage", {}).get("total_tokens", 0) +
                    final_response.get("usage", {}).get("total_tokens", 0)
                )
            },
            metadata={
                "self_critique": True,
                "initial_answer": initial_result.answer
            }
        )
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~1 token per 4 characters
        # For Chinese: ~1 token per 2 characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        
        return chinese_chars // 2 + other_chars // 4
    
    def compress_contexts(
        self,
        contexts: List[RetrievalResult],
        target_tokens: int
    ) -> List[RetrievalResult]:
        """Compress contexts to fit token limit.
        
        Args:
            contexts: Original contexts
            target_tokens: Target token count
            
        Returns:
            Compressed contexts
        """
        # Placeholder for context compression
        # In production, could use summarization or key phrase extraction
        return self._truncate_contexts(contexts, target_tokens)
    
    def select_examples(
        self,
        query: str,
        examples: List[Dict[str, str]],
        k: int = 2
    ) -> List[Dict[str, str]]:
        """Select relevant few-shot examples.
        
        Args:
            query: Current query
            examples: Available examples
            k: Number of examples to select
            
        Returns:
            Selected examples
        """
        # Simple selection based on query similarity
        # In production, could use embedding similarity
        
        selected = []
        query_terms = set(query.lower().split())
        
        # Score examples by term overlap
        scored_examples = []
        for example in examples:
            example_terms = set(example["query"].lower().split())
            overlap = len(query_terms & example_terms)
            scored_examples.append((overlap, example))
        
        # Sort by score and select top k
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        for _, example in scored_examples[:k]:
            selected.append(example)
        
        return selected