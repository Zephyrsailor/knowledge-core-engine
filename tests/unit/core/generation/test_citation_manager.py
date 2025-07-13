"""Unit tests for citation manager module."""

import pytest
from typing import List, Dict, Any

from knowledge_core_engine.core.generation.citation_manager import (
    CitationManager, Citation, CitationStyle
)
from knowledge_core_engine.core.retrieval.retriever import RetrievalResult


class TestCitationManager:
    """Test the CitationManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create CitationManager instance."""
        return CitationManager()
    
    @pytest.fixture
    def mock_contexts(self):
        """Create mock retrieval contexts."""
        return [
            RetrievalResult(
                chunk_id="chunk_1",
                content="RAG技术是结合检索和生成的方法。",
                score=0.95,
                metadata={
                    "document_title": "RAG技术详解",
                    "document_id": "doc_1",
                    "page": 5,
                    "author": "张三",
                    "year": "2024"
                }
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                content="向量数据库用于存储嵌入向量。",
                score=0.88,
                metadata={
                    "document_title": "向量数据库指南",
                    "document_id": "doc_2",
                    "page": 12,
                    "url": "https://example.com/vector-db"
                }
            ),
            RetrievalResult(
                chunk_id="chunk_3",
                content="大语言模型可以理解和生成自然语言。",
                score=0.82,
                metadata={
                    "document_title": "LLM基础",
                    "document_id": "doc_3",
                    "section": "2.1"
                }
            )
        ]
    
    def test_extract_citations_from_text(self, manager):
        """Test extracting citation markers from text."""
        text = """
        RAG技术是一种先进的方法[1]。它结合了检索[2]和生成[1]的优势。
        向量数据库[3]在其中扮演重要角色。
        """
        
        citations = manager.extract_citations(text)
        
        assert len(citations) == 4  # Including duplicate [1]
        assert citations == [1, 2, 1, 3]
    
    def test_map_citations_to_sources(self, manager, mock_contexts):
        """Test mapping citation indices to source documents."""
        text = "RAG技术[1]使用向量数据库[2]。"
        citations = [1, 2]
        
        mapped = manager.map_citations(
            citations=citations,
            contexts=mock_contexts
        )
        
        assert len(mapped) == 2
        assert mapped[1].chunk_id == "chunk_1"
        assert mapped[1].document_title == "RAG技术详解"
        assert mapped[2].chunk_id == "chunk_2"
        assert mapped[2].document_title == "向量数据库指南"
    
    def test_inline_citation_formatting(self, manager, mock_contexts):
        """Test inline citation formatting."""
        text = "RAG技术[1]和向量数据库[2]都很重要。"
        
        result = manager.format_citations(
            text=text,
            contexts=mock_contexts,
            style=CitationStyle.INLINE
        )
        
        assert "[1]" in result.text
        assert "[2]" in result.text
        assert len(result.citations) == 2
    
    def test_footnote_citation_formatting(self, manager, mock_contexts):
        """Test footnote citation formatting."""
        text = "RAG技术[1]是革命性的。"
        
        result = manager.format_citations(
            text=text,
            contexts=mock_contexts,
            style=CitationStyle.FOOTNOTE
        )
        
        assert "[1]" in result.text
        assert "---" in result.text or "参考文献" in result.text
        assert "RAG技术详解" in result.text
        assert "第5页" in result.text or "p.5" in result.text
    
    def test_endnote_citation_formatting(self, manager, mock_contexts):
        """Test endnote citation formatting."""
        text = "使用RAG[1]和LLM[3]构建系统。"
        
        result = manager.format_citations(
            text=text,
            contexts=mock_contexts,
            style=CitationStyle.ENDNOTE
        )
        
        assert "[1]" in result.text
        assert "[3]" in result.text
        # Endnotes should appear at the end
        assert result.text.rindex("参考文献") > result.text.index("[3]")
    
    def test_apa_style_formatting(self, manager, mock_contexts):
        """Test APA-style citation formatting."""
        text = "根据研究[1]，RAG技术很有效。"
        
        result = manager.format_citations(
            text=text,
            contexts=mock_contexts,
            style=CitationStyle.APA
        )
        
        # Should include author and year
        assert "张三" in result.text or "(2024)" in result.text
    
    def test_citation_deduplication(self, manager, mock_contexts):
        """Test handling duplicate citations."""
        text = "RAG[1]是重要的[1]技术[1]。"
        
        result = manager.format_citations(
            text=text,
            contexts=mock_contexts,
            style=CitationStyle.FOOTNOTE
        )
        
        # Should only have one entry for [1] in references
        ref_count = result.text.count("RAG技术详解")
        assert ref_count == 1
    
    def test_missing_citation_handling(self, manager, mock_contexts):
        """Test handling citations without matching contexts."""
        text = "引用不存在的来源[5]。"
        
        result = manager.format_citations(
            text=text,
            contexts=mock_contexts,
            style=CitationStyle.INLINE
        )
        
        # Should handle gracefully
        assert "[5]" in result.text
        # Should not crash, might add note or remove
    
    def test_url_citation_formatting(self, manager, mock_contexts):
        """Test formatting citations with URLs."""
        text = "向量数据库[2]的更多信息。"
        
        result = manager.format_citations(
            text=text,
            contexts=mock_contexts,
            style=CitationStyle.FOOTNOTE
        )
        
        assert "https://example.com/vector-db" in result.text
    
    def test_custom_citation_template(self, manager, mock_contexts):
        """Test using custom citation templates."""
        text = "参考文献[1]。"
        
        custom_template = "{author} ({year}). {title}. 第{page}页"
        
        result = manager.format_citations(
            text=text,
            contexts=mock_contexts,
            style=CitationStyle.CUSTOM,
            template=custom_template
        )
        
        assert "张三 (2024)" in result.text
        assert "RAG技术详解" in result.text
        assert "第5页" in result.text
    
    def test_citation_numbering_consistency(self, manager, mock_contexts):
        """Test consistent citation numbering."""
        text = "首先是RAG[1]，然后是数据库[2]，最后回到RAG[1]。"
        
        result = manager.format_citations(
            text=text,
            contexts=mock_contexts
        )
        
        # Same source should have same number
        first_rag = result.text.index("[1]")
        last_rag = result.text.rindex("[1]")
        assert first_rag < last_rag
        
        # Only 2 unique citations
        assert len(result.citations) == 2
    
    def test_generate_bibliography(self, manager, mock_contexts):
        """Test generating bibliography from contexts."""
        bibliography = manager.generate_bibliography(
            contexts=mock_contexts,
            style=CitationStyle.APA
        )
        
        assert len(bibliography) == 3
        assert "RAG技术详解" in bibliography[0]
        assert "向量数据库指南" in bibliography[1]
        assert "LLM基础" in bibliography[2]


class TestCitation:
    """Test the Citation class."""
    
    def test_citation_creation(self):
        """Test creating a citation object."""
        citation = Citation(
            index=1,
            chunk_id="chunk_123",
            document_title="Test Document",
            document_id="doc_123",
            page=10,
            section="2.1",
            author="John Doe",
            year="2024",
            url="https://example.com"
        )
        
        assert citation.index == 1
        assert citation.chunk_id == "chunk_123"
        assert citation.document_title == "Test Document"
        assert citation.page == 10
        assert citation.author == "John Doe"
    
    def test_citation_from_retrieval_result(self):
        """Test creating citation from retrieval result."""
        result = RetrievalResult(
            chunk_id="chunk_1",
            content="Content",
            score=0.9,
            metadata={
                "document_title": "Title",
                "page": 5,
                "author": "Author"
            }
        )
        
        citation = Citation.from_retrieval_result(result, index=1)
        
        assert citation.index == 1
        assert citation.chunk_id == "chunk_1"
        assert citation.document_title == "Title"
        assert citation.page == 5
        assert citation.author == "Author"
    
    def test_citation_formatting_methods(self):
        """Test various citation formatting methods."""
        citation = Citation(
            index=1,
            chunk_id="chunk_1",
            document_title="AI Handbook",
            page=25,
            author="Jane Smith",
            year="2024"
        )
        
        # Inline format
        inline = citation.format_inline()
        assert inline == "[1]"
        
        # Short format
        short = citation.format_short()
        assert "AI Handbook" in short
        assert "p.25" in short or "页25" in short
        
        # Full format
        full = citation.format_full()
        assert "Jane Smith" in full
        assert "2024" in full
        assert "AI Handbook" in full
        
        # APA format
        apa = citation.format_apa()
        assert "Smith, J." in apa or "Jane Smith" in apa
        assert "(2024)" in apa
    
    def test_citation_equality(self):
        """Test citation equality comparison."""
        citation1 = Citation(
            index=1,
            chunk_id="chunk_1",
            document_title="Title"
        )
        
        citation2 = Citation(
            index=1,
            chunk_id="chunk_1",
            document_title="Title"
        )
        
        citation3 = Citation(
            index=2,
            chunk_id="chunk_2",
            document_title="Title"
        )
        
        assert citation1 == citation2
        assert citation1 != citation3


class TestCitationStyle:
    """Test CitationStyle enum and utilities."""
    
    def test_citation_style_values(self):
        """Test citation style enum values."""
        assert CitationStyle.INLINE.value == "inline"
        assert CitationStyle.FOOTNOTE.value == "footnote"
        assert CitationStyle.ENDNOTE.value == "endnote"
        assert CitationStyle.APA.value == "apa"
        assert CitationStyle.MLA.value == "mla"
        assert CitationStyle.CHICAGO.value == "chicago"
        assert CitationStyle.CUSTOM.value == "custom"
    
    def test_style_from_string(self):
        """Test creating style from string."""
        style = CitationStyle.from_string("footnote")
        assert style == CitationStyle.FOOTNOTE
        
        style = CitationStyle.from_string("APA")
        assert style == CitationStyle.APA
        
        # Default for unknown
        style = CitationStyle.from_string("unknown")
        assert style == CitationStyle.INLINE
    
    def test_style_descriptions(self):
        """Test getting style descriptions."""
        desc = CitationStyle.get_description(CitationStyle.FOOTNOTE)
        assert "脚注" in desc or "footnote" in desc.lower()
        
        desc = CitationStyle.get_description(CitationStyle.APA)
        assert "APA" in desc


class TestAdvancedCitationFeatures:
    """Test advanced citation features."""
    
    @pytest.fixture
    def manager(self):
        """Create citation manager with advanced features."""
        return CitationManager(
            enable_smart_grouping=True,
            enable_url_shortening=True
        )
    
    def test_citation_grouping(self, manager):
        """Test grouping multiple citations."""
        text = "多项研究[1][2][3]表明RAG很有效。"
        
        grouped = manager.group_citations(text)
        
        # Should group consecutive citations
        assert "[1-3]" in grouped or "[1,2,3]" in grouped
    
    def test_citation_smart_placement(self, manager):
        """Test smart citation placement."""
        text = "RAG技术很重要"
        source_text = "RAG技术是结合检索和生成的方法"
        
        # Should suggest where to place citation
        suggested = manager.suggest_citation_placement(
            text=text,
            source_text=source_text
        )
        
        assert "[1]" in suggested
        assert suggested.index("[1]") > suggested.index("RAG技术")
    
    def test_citation_conflict_resolution(self, manager):
        """Test resolving citation conflicts."""
        # When merging documents with existing citations
        text1 = "第一个文档引用[1]和[2]。"
        text2 = "第二个文档引用[1]和[3]。"
        
        merged = manager.merge_citations(
            texts=[text1, text2],
            contexts_list=[[], []]  # Mock contexts
        )
        
        # Should renumber to avoid conflicts
        assert "[3]" in merged or "[4]" in merged
    
    def test_citation_validation(self, manager):
        """Test citation validation."""
        text = "有效引用[1]和无效引用[99]。"
        contexts = [
            RetrievalResult("chunk_1", "Content", 0.9, {})
        ]
        
        issues = manager.validate_citations(text, contexts)
        
        assert len(issues) > 0
        assert any("99" in issue for issue in issues)