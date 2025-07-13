"""Prompt builder for generation module."""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from ..config import RAGConfig
from ..retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Prompt template definition."""
    name: str
    template: str
    description: Optional[str] = None
    variables: List[str] = None
    
    def __post_init__(self):
        if self.variables is None:
            # Extract variables from template
            import re
            self.variables = re.findall(r'\{(\w+)\}', self.template)
    
    def format(self, **kwargs) -> str:
        """Format template with variables."""
        return self.template.format(**kwargs)
    
    def is_valid(self) -> bool:
        """Check if template is valid."""
        required = {"query", "contexts"}
        return required.issubset(set(self.variables))
    
    @classmethod
    def get_builtin_templates(cls) -> Dict[str, 'PromptTemplate']:
        """Get built-in templates."""
        return {
            "qa_basic": cls(
                name="qa_basic",
                template="""根据以下上下文信息回答问题。

问题：{query}

上下文信息：
{contexts}

请基于上述上下文信息回答问题。如果上下文中没有相关信息，请说明。""",
                description="Basic Q&A template"
            ),
            
            "qa_with_citations": cls(
                name="qa_with_citations",
                template="""根据以下上下文信息回答问题，并在答案中标注引用来源。

问题：{query}

上下文信息：
{contexts}

要求：
1. 基于上下文信息准确回答问题
2. 在使用某个文档的信息时，用[数字]标注，如[1]、[2]
3. 确保引用编号与上下文文档编号对应
4. 如果信息来自多个文档，分别标注

请回答：""",
                description="Q&A with citations"
            ),
            
            "qa_structured": cls(
                name="qa_structured",
                template="""根据以下上下文信息，以结构化的方式回答问题。

问题：{query}

上下文信息：
{contexts}

请按以下格式组织答案：
1. 简要回答（1-2句话）
2. 详细说明（如需要）
3. 相关要点（如适用）

回答：""",
                description="Structured answer template"
            ),
            
            "qa_cot": cls(
                name="qa_cot",
                template="""让我们一步步思考并回答这个问题。

问题：{query}

上下文信息：
{contexts}

思考步骤：
1. 理解问题的核心需求
2. 从上下文中找出相关信息
3. 分析这些信息如何回答问题
4. 组织答案

基于以上思考，我的回答是：""",
                description="Chain of thought template"
            )
        }


class ContextFormatter:
    """Format contexts for prompt."""
    
    def format_contexts(
        self,
        contexts: List[RetrievalResult],
        include_indices: bool = True,
        include_metadata: bool = True,
        compact: bool = False,
        format: str = "text",
        separator: str = "\n\n",
        max_length: Optional[int] = None
    ) -> str:
        """Format contexts for inclusion in prompt.
        
        Args:
            contexts: List of retrieval results
            include_indices: Whether to include document indices
            include_metadata: Whether to include metadata
            compact: Use compact formatting
            format: Output format (text, markdown, json)
            separator: Separator between contexts
            max_length: Maximum total length
            
        Returns:
            Formatted contexts string
        """
        if format == "json":
            return self._format_json(contexts)
        elif format == "markdown":
            return self._format_markdown(contexts, include_indices, include_metadata)
        else:
            return self._format_text(
                contexts, include_indices, include_metadata,
                compact, separator, max_length
            )
    
    def _format_text(
        self,
        contexts: List[RetrievalResult],
        include_indices: bool,
        include_metadata: bool,
        compact: bool,
        separator: str,
        max_length: Optional[int]
    ) -> str:
        """Format as plain text."""
        formatted_contexts = []
        
        for i, context in enumerate(contexts):
            parts = []
            
            # Document index
            if include_indices:
                parts.append(f"[文档{i+1}]")
            
            # Metadata
            if include_metadata:
                meta_parts = []
                if context.metadata.get("document_title"):
                    meta_parts.append(f"来源：{context.metadata['document_title']}")
                if context.metadata.get("page"):
                    meta_parts.append(f"第{context.metadata['page']}页")
                if context.metadata.get("chunk_id"):
                    meta_parts.append(f"ID: {context.chunk_id}")
                
                if meta_parts:
                    parts.append(" | ".join(meta_parts))
            
            # Content
            if compact:
                # Single line format
                content_preview = context.content.replace('\n', ' ')[:200]
                if len(context.content) > 200:
                    content_preview += "..."
                parts.append(content_preview)
                formatted = " - ".join(parts)
            else:
                # Multi-line format
                if parts:
                    header = " ".join(parts)
                    formatted = f"{header}\n{context.content}"
                else:
                    formatted = context.content
            
            formatted_contexts.append(formatted)
        
        result = separator.join(formatted_contexts)
        
        # Truncate if needed
        if max_length and len(result) > max_length:
            result = result[:max_length-3] + "..."
        
        return result
    
    def _format_markdown(
        self,
        contexts: List[RetrievalResult],
        include_indices: bool,
        include_metadata: bool
    ) -> str:
        """Format as markdown."""
        formatted = []
        
        for i, context in enumerate(contexts):
            # Header
            if include_indices:
                formatted.append(f"## 文档 {i+1}")
            
            # Metadata table
            if include_metadata and context.metadata:
                formatted.append("\n| 属性 | 值 |")
                formatted.append("|------|-----|")
                for key, value in context.metadata.items():
                    if value:
                        formatted.append(f"| {key} | {value} |")
                formatted.append("")
            
            # Content
            formatted.append("```")
            formatted.append(context.content)
            formatted.append("```\n")
        
        return "\n".join(formatted)
    
    def _format_json(self, contexts: List[RetrievalResult]) -> str:
        """Format as JSON."""
        data = []
        for context in contexts:
            data.append({
                "chunk_id": context.chunk_id,
                "content": context.content,
                "score": context.score,
                "metadata": context.metadata
            })
        return json.dumps(data, ensure_ascii=False, indent=2)


class PromptBuilder:
    """Build prompts for generation."""
    
    def __init__(self, config: RAGConfig):
        """Initialize prompt builder.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        self.formatter = ContextFormatter()
        self.templates = PromptTemplate.get_builtin_templates()
    
    def build_prompt(
        self,
        query: str,
        contexts: List[RetrievalResult],
        template: Optional[str] = None,
        include_citations: bool = False,
        output_format: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        enable_cot: bool = False,
        **kwargs
    ) -> str:
        """Build prompt for generation.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            template: Template name or custom template
            include_citations: Whether to include citation instructions
            output_format: Desired output format
            few_shot_examples: Few-shot examples
            enable_cot: Enable chain of thought
            **kwargs: Additional parameters
            
        Returns:
            Complete prompt
        """
        # Select template
        if enable_cot:
            template_obj = self.templates["qa_cot"]
        elif include_citations:
            template_obj = self.templates["qa_with_citations"]
        elif output_format == "structured":
            template_obj = self.templates["qa_structured"]
        elif isinstance(template, str) and template in self.templates:
            template_obj = self.templates[template]
        elif isinstance(template, str):
            # Custom template string
            template_obj = PromptTemplate("custom", template)
        else:
            template_obj = self.templates["qa_basic"]
        
        # Format contexts
        formatted_contexts = self.formatter.format_contexts(
            contexts,
            include_indices=True,
            include_metadata=True,
            compact=False
        )
        
        # Add few-shot examples if provided
        if few_shot_examples:
            examples_text = self._format_examples(few_shot_examples)
            formatted_contexts = f"{examples_text}\n\n当前问题的上下文：\n{formatted_contexts}"
        
        # Build prompt
        prompt = template_obj.format(
            query=query,
            contexts=formatted_contexts
        )
        
        # Add output format instructions
        if output_format == "structured" and "结构" not in prompt:
            prompt += "\n\n请以结构化的方式组织你的回答，使用标题、列表等格式。"
        
        return prompt
    
    def build_messages(
        self,
        query: Optional[str] = None,
        contexts: Optional[List[RetrievalResult]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build messages for chat-based LLMs.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            prompt: Pre-built prompt
            system_prompt: System prompt
            
        Returns:
            List of messages
        """
        messages = []
        
        # System prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            messages.append({
                "role": "system",
                "content": self._get_default_system_prompt()
            })
        
        # User message
        if prompt:
            user_content = prompt
        elif query and contexts is not None:
            user_content = self.build_prompt(query, contexts)
        else:
            raise ValueError("Either prompt or (query, contexts) must be provided")
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def build_no_context_prompt(self, query: str) -> str:
        """Build prompt for no context scenario.
        
        Args:
            query: User query
            
        Returns:
            No context prompt
        """
        return f"""用户提出了以下问题，但我没有找到相关的文档信息：

问题：{query}

请礼貌地说明没有找到相关信息，并建议用户：
1. 尝试使用不同的关键词
2. 提供更多上下文
3. 确认问题是否在知识库范围内

回复："""
    
    def build_critique_prompt(
        self,
        query: str,
        answer: str,
        contexts: List[RetrievalResult]
    ) -> str:
        """Build prompt for self-critique.
        
        Args:
            query: Original query
            answer: Generated answer
            contexts: Source contexts
            
        Returns:
            Critique prompt
        """
        formatted_contexts = self.formatter.format_contexts(
            contexts, include_indices=True, compact=True
        )
        
        return f"""请评估以下回答的质量：

原始问题：{query}

生成的回答：
{answer}

参考上下文：
{formatted_contexts}

请从以下方面评估：
1. 准确性：回答是否准确基于上下文
2. 完整性：是否充分回答了问题
3. 清晰度：表达是否清晰易懂
4. 引用：引用是否正确（如适用）

请指出需要改进的地方："""
    
    def build_improvement_prompt(
        self,
        query: str,
        answer: str,
        critique: str,
        contexts: List[RetrievalResult]
    ) -> str:
        """Build prompt for improvement based on critique.
        
        Args:
            query: Original query
            answer: Original answer
            critique: Critique feedback
            contexts: Source contexts
            
        Returns:
            Improvement prompt
        """
        formatted_contexts = self.formatter.format_contexts(contexts)
        
        return f"""基于以下反馈改进回答：

原始问题：{query}

原始回答：
{answer}

改进建议：
{critique}

参考上下文：
{formatted_contexts}

请提供改进后的回答："""
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        lang = self.config.extra_params.get("language", "zh")
        
        if lang == "zh":
            return """你是一个专业的知识助手。请基于提供的上下文信息准确回答用户问题。

要求：
1. 答案必须基于提供的上下文，不要编造信息
2. 如果上下文中没有相关信息，请明确说明
3. 保持专业、客观的语气
4. 适当组织答案结构，使其清晰易读"""
        else:
            return """You are a professional knowledge assistant. Please answer user questions accurately based on the provided context.

Requirements:
1. Answers must be based on the provided context, do not make up information
2. If there is no relevant information in the context, please state so clearly
3. Maintain a professional and objective tone
4. Organize your answer appropriately for clarity"""
    
    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples."""
        formatted = "以下是一些问答示例：\n\n"
        
        for i, example in enumerate(examples):
            formatted += f"示例 {i+1}:\n"
            formatted += f"问题：{example['query']}\n"
            formatted += f"回答：{example['answer']}\n\n"
        
        return formatted.strip()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count.
        
        Args:
            text: Input text
            
        Returns:
            Estimated tokens
        """
        # Simple estimation
        import re
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
        # Simple truncation for now
        compressed = []
        current_tokens = 0
        
        for context in contexts:
            estimated = self.estimate_tokens(context.content)
            if current_tokens + estimated > target_tokens:
                # Truncate this context
                remaining = target_tokens - current_tokens
                chars_to_keep = remaining * 3  # Rough estimate
                
                truncated = RetrievalResult(
                    chunk_id=context.chunk_id,
                    content=context.content[:chars_to_keep] + "...",
                    score=context.score,
                    metadata=context.metadata
                )
                compressed.append(truncated)
                break
            else:
                compressed.append(context)
                current_tokens += estimated
        
        return compressed
    
    def select_examples(
        self,
        query: str,
        examples: List[Dict[str, str]],
        k: int = 2
    ) -> List[Dict[str, str]]:
        """Select relevant examples.
        
        Args:
            query: Current query
            examples: Available examples
            k: Number to select
            
        Returns:
            Selected examples
        """
        # Simple keyword matching
        query_terms = set(query.lower().split())
        
        scored = []
        for example in examples:
            example_terms = set(example["query"].lower().split())
            score = len(query_terms & example_terms)
            scored.append((score, example))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [ex for _, ex in scored[:k]]