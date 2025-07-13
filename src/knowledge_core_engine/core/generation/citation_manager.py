"""Citation management for generated answers."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..retrieval.retriever import RetrievalResult


class CitationStyle(Enum):
    """Citation formatting styles."""
    INLINE = "inline"          # [1], [2]
    FOOTNOTE = "footnote"      # With footnotes at bottom
    ENDNOTE = "endnote"        # With endnotes
    APA = "apa"                # APA style
    MLA = "mla"                # MLA style
    CHICAGO = "chicago"        # Chicago style
    CUSTOM = "custom"          # Custom format
    
    @classmethod
    def from_string(cls, value: str) -> 'CitationStyle':
        """Create from string value."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.INLINE
    
    @classmethod
    def get_description(cls, style: 'CitationStyle') -> str:
        """Get style description."""
        descriptions = {
            cls.INLINE: "内联引用格式 [1]",
            cls.FOOTNOTE: "脚注格式，引用详情在底部",
            cls.ENDNOTE: "尾注格式，引用详情在文末",
            cls.APA: "APA学术引用格式",
            cls.MLA: "MLA学术引用格式",
            cls.CHICAGO: "芝加哥引用格式",
            cls.CUSTOM: "自定义引用格式"
        }
        return descriptions.get(style, "")


@dataclass
class Citation:
    """Citation information."""
    index: int
    chunk_id: str
    document_title: Optional[str] = None
    document_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None
    text: Optional[str] = None
    
    @classmethod
    def from_retrieval_result(
        cls,
        result: RetrievalResult,
        index: int
    ) -> 'Citation':
        """Create from retrieval result."""
        metadata = result.metadata or {}
        
        return cls(
            index=index,
            chunk_id=result.chunk_id,
            document_title=metadata.get("document_title"),
            document_id=metadata.get("document_id"),
            page=metadata.get("page"),
            section=metadata.get("section"),
            author=metadata.get("author"),
            year=metadata.get("year"),
            url=metadata.get("url"),
            text=result.content[:200] + "..." if len(result.content) > 200 else result.content
        )
    
    def format_inline(self) -> str:
        """Format as inline citation."""
        return f"[{self.index}]"
    
    def format_short(self) -> str:
        """Format as short citation."""
        parts = []
        if self.document_title:
            parts.append(self.document_title)
        if self.page:
            parts.append(f"p.{self.page}")
        
        return f"[{self.index}] " + ", ".join(parts) if parts else f"[{self.index}]"
    
    def format_full(self) -> str:
        """Format as full citation."""
        parts = []
        
        if self.author:
            parts.append(self.author)
        if self.year:
            parts.append(f"({self.year})")
        if self.document_title:
            parts.append(f"《{self.document_title}》")
        if self.page:
            parts.append(f"第{self.page}页")
        if self.section:
            parts.append(f"第{self.section}节")
        if self.url:
            parts.append(f"链接：{self.url}")
        
        return f"[{self.index}] " + ". ".join(parts)
    
    def format_apa(self) -> str:
        """Format in APA style."""
        if self.author and self.year:
            return f"{self.author} ({self.year})"
        elif self.document_title and self.year:
            return f"{self.document_title} ({self.year})"
        else:
            return self.format_short()
    
    def __eq__(self, other) -> bool:
        """Check equality based on chunk_id."""
        if not isinstance(other, Citation):
            return False
        return self.chunk_id == other.chunk_id


@dataclass
class CitationResult:
    """Result of citation formatting."""
    text: str
    citations: List[Citation]
    style: CitationStyle


class CitationManager:
    """Manage citations in generated text."""
    
    def __init__(
        self,
        enable_smart_grouping: bool = False,
        enable_url_shortening: bool = False
    ):
        """Initialize citation manager.
        
        Args:
            enable_smart_grouping: Enable grouping consecutive citations
            enable_url_shortening: Enable URL shortening
        """
        self.enable_smart_grouping = enable_smart_grouping
        self.enable_url_shortening = enable_url_shortening
    
    def extract_citations(self, text: str) -> List[int]:
        """Extract citation indices from text.
        
        Args:
            text: Text containing citations like [1], [2]
            
        Returns:
            List of citation indices
        """
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]
    
    def map_citations(
        self,
        citations: List[int],
        contexts: List[RetrievalResult]
    ) -> Dict[int, Citation]:
        """Map citation indices to sources.
        
        Args:
            citations: List of citation indices
            contexts: Source contexts
            
        Returns:
            Mapping of index to Citation
        """
        mapped = {}
        
        for idx in set(citations):  # Unique indices
            if 1 <= idx <= len(contexts):
                context = contexts[idx - 1]  # 1-based to 0-based
                mapped[idx] = Citation.from_retrieval_result(context, idx)
        
        return mapped
    
    def format_citations(
        self,
        text: str,
        contexts: List[RetrievalResult],
        style: CitationStyle = CitationStyle.INLINE,
        template: Optional[str] = None
    ) -> CitationResult:
        """Format citations in text.
        
        Args:
            text: Text with citation markers
            contexts: Source contexts
            style: Citation style
            template: Custom template (for CUSTOM style)
            
        Returns:
            Formatted citation result
        """
        # Extract citations
        citation_indices = self.extract_citations(text)
        
        # Map to sources
        citation_map = self.map_citations(citation_indices, contexts)
        
        # Format based on style
        if style == CitationStyle.INLINE:
            return self._format_inline(text, citation_map)
        elif style == CitationStyle.FOOTNOTE:
            return self._format_footnote(text, citation_map)
        elif style == CitationStyle.ENDNOTE:
            return self._format_endnote(text, citation_map)
        elif style == CitationStyle.APA:
            return self._format_apa(text, citation_map)
        elif style == CitationStyle.CUSTOM and template:
            return self._format_custom(text, citation_map, template)
        else:
            return self._format_inline(text, citation_map)
    
    def _format_inline(
        self,
        text: str,
        citations: Dict[int, Citation]
    ) -> CitationResult:
        """Format with inline citations."""
        # Just keep the original [n] markers
        citation_list = list(citations.values())
        
        return CitationResult(
            text=text,
            citations=citation_list,
            style=CitationStyle.INLINE
        )
    
    def _format_footnote(
        self,
        text: str,
        citations: Dict[int, Citation]
    ) -> CitationResult:
        """Format with footnotes."""
        # Add footnotes at the bottom
        formatted_text = text
        
        if citations:
            formatted_text += "\n\n---\n参考文献：\n"
            
            for idx in sorted(citations.keys()):
                citation = citations[idx]
                formatted_text += f"{citation.format_full()}\n"
        
        return CitationResult(
            text=formatted_text,
            citations=list(citations.values()),
            style=CitationStyle.FOOTNOTE
        )
    
    def _format_endnote(
        self,
        text: str,
        citations: Dict[int, Citation]
    ) -> CitationResult:
        """Format with endnotes."""
        # Similar to footnotes but with different formatting
        formatted_text = text
        
        if citations:
            formatted_text += "\n\n========== 参考文献 ==========\n"
            
            for idx in sorted(citations.keys()):
                citation = citations[idx]
                formatted_text += f"\n{citation.format_full()}\n"
        
        return CitationResult(
            text=formatted_text,
            citations=list(citations.values()),
            style=CitationStyle.ENDNOTE
        )
    
    def _format_apa(
        self,
        text: str,
        citations: Dict[int, Citation]
    ) -> CitationResult:
        """Format in APA style."""
        formatted_text = text
        
        # Replace [n] with (Author, Year)
        for idx, citation in citations.items():
            old_marker = f"[{idx}]"
            new_marker = citation.format_apa()
            formatted_text = formatted_text.replace(old_marker, new_marker)
        
        # Add references section
        if citations:
            formatted_text += "\n\nReferences:\n"
            for citation in sorted(citations.values(), key=lambda c: c.author or c.document_title or ""):
                formatted_text += f"{citation.format_full()}\n"
        
        return CitationResult(
            text=formatted_text,
            citations=list(citations.values()),
            style=CitationStyle.APA
        )
    
    def _format_custom(
        self,
        text: str,
        citations: Dict[int, Citation],
        template: str
    ) -> CitationResult:
        """Format with custom template."""
        formatted_text = text
        
        # Apply custom template
        for idx, citation in citations.items():
            # Replace template variables
            formatted_citation = template.format(
                index=citation.index,
                author=citation.author or "Unknown",
                year=citation.year or "n.d.",
                title=citation.document_title or "Untitled",
                page=citation.page or "",
                url=citation.url or ""
            )
            
            # Replace in text
            old_marker = f"[{idx}]"
            formatted_text = formatted_text.replace(old_marker, formatted_citation)
        
        return CitationResult(
            text=formatted_text,
            citations=list(citations.values()),
            style=CitationStyle.CUSTOM
        )
    
    def generate_bibliography(
        self,
        contexts: List[RetrievalResult],
        style: CitationStyle = CitationStyle.APA
    ) -> List[str]:
        """Generate bibliography from contexts.
        
        Args:
            contexts: Source contexts
            style: Citation style
            
        Returns:
            List of formatted citations
        """
        bibliography = []
        
        for i, context in enumerate(contexts):
            citation = Citation.from_retrieval_result(context, i + 1)
            
            if style == CitationStyle.APA:
                bibliography.append(citation.format_full())
            else:
                bibliography.append(citation.format_full())
        
        return bibliography
    
    def group_citations(self, text: str) -> str:
        """Group consecutive citations.
        
        Args:
            text: Text with citations like [1][2][3]
            
        Returns:
            Text with grouped citations like [1-3]
        """
        # Find consecutive citation patterns
        pattern = r'(\[\d+\](?:\[\d+\])+)'
        
        def replace_group(match):
            group = match.group(0)
            # Extract numbers
            numbers = [int(n) for n in re.findall(r'\d+', group)]
            
            if len(numbers) > 2 and numbers == list(range(numbers[0], numbers[-1] + 1)):
                # Consecutive range
                return f"[{numbers[0]}-{numbers[-1]}]"
            else:
                # Non-consecutive, use comma
                return "[" + ",".join(str(n) for n in numbers) + "]"
        
        return re.sub(pattern, replace_group, text)
    
    def suggest_citation_placement(
        self,
        text: str,
        source_text: str,
        threshold: float = 0.3
    ) -> str:
        """Suggest where to place citations.
        
        Args:
            text: Generated text without citations
            source_text: Source context text
            threshold: Similarity threshold
            
        Returns:
            Text with suggested citation placements
        """
        # Simple implementation: add citation after sentences that closely match source
        sentences = re.split(r'[。！？\.]', text)
        source_keywords = set(source_text.lower().split())
        
        result_parts = []
        citation_added = False
        
        for sentence in sentences:
            if sentence.strip():
                sentence_keywords = set(sentence.lower().split())
                overlap = len(sentence_keywords & source_keywords)
                
                if overlap / max(len(sentence_keywords), 1) > threshold:
                    result_parts.append(sentence + "[1]")
                    citation_added = True
                else:
                    result_parts.append(sentence)
        
        # If no citations added, add at the end
        if not citation_added and result_parts:
            result_parts[-1] += "[1]"
        
        return "。".join(result_parts) if any("。" in text for text in result_parts) else ". ".join(result_parts)
    
    def merge_citations(
        self,
        texts: List[str],
        contexts_list: List[List[RetrievalResult]]
    ) -> str:
        """Merge multiple texts with citations.
        
        Args:
            texts: List of texts with citations
            contexts_list: Corresponding contexts for each text
            
        Returns:
            Merged text with renumbered citations
        """
        merged_text = ""
        current_offset = 0
        all_contexts = []
        
        for text, contexts in zip(texts, contexts_list):
            # Renumber citations in this text
            renumbered = self._renumber_citations(text, current_offset)
            merged_text += renumbered + "\n\n"
            
            all_contexts.extend(contexts)
            current_offset += len(contexts)
        
        return merged_text.strip()
    
    def _renumber_citations(self, text: str, offset: int) -> str:
        """Renumber citations with offset.
        
        Args:
            text: Text with citations
            offset: Number to add to each citation
            
        Returns:
            Text with renumbered citations
        """
        def replace_citation(match):
            old_num = int(match.group(1))
            new_num = old_num + offset
            return f"[{new_num}]"
        
        return re.sub(r'\[(\d+)\]', replace_citation, text)
    
    def validate_citations(
        self,
        text: str,
        contexts: List[RetrievalResult]
    ) -> List[str]:
        """Validate citations in text.
        
        Args:
            text: Text with citations
            contexts: Available contexts
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Extract citations
        citations = self.extract_citations(text)
        
        # Check for out-of-range citations
        for idx in citations:
            if idx < 1 or idx > len(contexts):
                issues.append(f"Citation [{idx}] is out of range (1-{len(contexts)})")
        
        # Check for missing citations
        if self.enable_smart_grouping:
            # Check if important facts lack citations
            # This is a simplified check
            important_patterns = [
                r'研究表明',
                r'根据.*报告',
                r'数据显示',
                r'实验证明'
            ]
            
            for pattern in important_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Check if citation follows within 50 chars
                    following_text = text[match.end():match.end() + 50]
                    if not re.search(r'\[\d+\]', following_text):
                        issues.append(f"Statement '{match.group()}' may need citation")
        
        return issues