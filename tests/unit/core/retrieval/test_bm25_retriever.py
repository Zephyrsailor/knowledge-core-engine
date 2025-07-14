"""Tests for BM25 retriever."""

import pytest
from knowledge_core_engine.core.retrieval.bm25_retriever import BM25Retriever


class TestBM25Retriever:
    """Test BM25 retriever functionality."""
    
    @pytest.fixture
    def bm25_retriever(self):
        """Create BM25 retriever instance."""
        return BM25Retriever(k1=1.5, b=0.75)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                'id': 'doc1',
                'content': 'RAG技术是检索增强生成的简称，它结合了检索和生成两种方法',
                'metadata': {'source': 'test1.md', 'type': 'definition'}
            },
            {
                'id': 'doc2',
                'content': '向量数据库用于存储和检索嵌入向量，支持相似度搜索',
                'metadata': {'source': 'test2.md', 'type': 'technical'}
            },
            {
                'id': 'doc3',
                'content': '大语言模型可以理解和生成人类语言，是AI技术的重要突破',
                'metadata': {'source': 'test3.md', 'type': 'definition'}
            },
            {
                'id': 'doc4',
                'content': 'BM25是一种经典的文本检索算法，基于词频和逆文档频率',
                'metadata': {'source': 'test4.md', 'type': 'algorithm'}
            }
        ]
    
    def test_initialization(self, bm25_retriever):
        """Test BM25 retriever initialization."""
        assert bm25_retriever.k1 == 1.5
        assert bm25_retriever.b == 0.75
        assert bm25_retriever.epsilon == 0.25
        assert len(bm25_retriever.documents) == 0
        assert bm25_retriever.avgdl == 0
    
    def test_add_documents(self, bm25_retriever, sample_documents):
        """Test adding documents to the index."""
        bm25_retriever.add_documents(sample_documents)
        
        assert len(bm25_retriever.documents) == 4
        assert len(bm25_retriever.doc_ids) == 4
        assert bm25_retriever.avgdl > 0
        assert 'doc1' in bm25_retriever.documents
        assert 'doc1' in bm25_retriever.doc_metadata
        
        # Check IDF scores are calculated
        assert len(bm25_retriever.idf) > 0
    
    def test_search_basic(self, bm25_retriever, sample_documents):
        """Test basic search functionality."""
        bm25_retriever.add_documents(sample_documents)
        
        # Search for RAG
        results = bm25_retriever.search("RAG技术", top_k=2)
        
        assert len(results) <= 2
        assert results[0]['id'] == 'doc1'  # Should match the RAG document
        assert results[0]['score'] > 0
        assert 'content' in results[0]
        assert 'metadata' in results[0]
    
    def test_search_with_filters(self, bm25_retriever, sample_documents):
        """Test search with metadata filters."""
        bm25_retriever.add_documents(sample_documents)
        
        # Search with type filter
        results = bm25_retriever.search(
            "技术",
            top_k=10,
            filters={'type': 'definition'}
        )
        
        # Should only return documents with type='definition'
        for result in results:
            assert result['metadata']['type'] == 'definition'
    
    def test_search_empty_index(self, bm25_retriever):
        """Test search on empty index."""
        results = bm25_retriever.search("test", top_k=5)
        assert results == []
    
    def test_chinese_tokenization(self, bm25_retriever):
        """Test Chinese text tokenization."""
        docs = [
            {'id': 'doc1', 'content': '自然语言处理是人工智能的重要分支'},
            {'id': 'doc2', 'content': '深度学习推动了自然语言处理的发展'}
        ]
        bm25_retriever.add_documents(docs)
        
        # Search for a term that appears in both documents
        results = bm25_retriever.search("自然语言处理", top_k=2)
        
        assert len(results) == 2
        # Both documents should be returned as they contain the search term
        doc_ids = [r['id'] for r in results]
        assert 'doc1' in doc_ids
        assert 'doc2' in doc_ids
    
    def test_update_document(self, bm25_retriever, sample_documents):
        """Test updating a document."""
        bm25_retriever.add_documents(sample_documents[:2])
        
        # Update doc1
        bm25_retriever.update_document(
            'doc1',
            'RAG技术的新描述，包含更多关于检索增强生成的细节',
            {'source': 'updated.md', 'type': 'updated'}
        )
        
        # Search should return updated content
        results = bm25_retriever.search("RAG技术", top_k=1)
        assert len(results) > 0
        assert '新描述' in results[0]['content']
        assert results[0]['metadata']['type'] == 'updated'
    
    def test_remove_document(self, bm25_retriever, sample_documents):
        """Test removing a document."""
        bm25_retriever.add_documents(sample_documents)
        initial_count = len(bm25_retriever.documents)
        
        # Remove doc1
        bm25_retriever.remove_document('doc1')
        
        assert len(bm25_retriever.documents) == initial_count - 1
        assert 'doc1' not in bm25_retriever.documents
        assert 'doc1' not in bm25_retriever.doc_ids
        
        # Search should not return removed document
        results = bm25_retriever.search("RAG技术", top_k=10)
        doc_ids = [r['id'] for r in results]
        assert 'doc1' not in doc_ids
    
    def test_clear(self, bm25_retriever, sample_documents):
        """Test clearing all documents."""
        bm25_retriever.add_documents(sample_documents)
        bm25_retriever.clear()
        
        assert len(bm25_retriever.documents) == 0
        assert len(bm25_retriever.doc_ids) == 0
        assert len(bm25_retriever.idf) == 0
        assert bm25_retriever.avgdl == 0
    
    def test_get_stats(self, bm25_retriever, sample_documents):
        """Test getting index statistics."""
        bm25_retriever.add_documents(sample_documents)
        stats = bm25_retriever.get_stats()
        
        assert stats['num_documents'] == 4
        assert stats['num_terms'] > 0
        assert stats['avg_doc_length'] > 0
        assert stats['k1'] == 1.5
        assert stats['b'] == 0.75
    
    def test_score_calculation(self, bm25_retriever):
        """Test BM25 score calculation."""
        # Add documents with known content
        docs = [
            {'id': 'doc1', 'content': 'apple apple apple'},
            {'id': 'doc2', 'content': 'apple banana'},
            {'id': 'doc3', 'content': 'banana banana banana'}
        ]
        bm25_retriever.add_documents(docs)
        
        # Search for "apple"
        results = bm25_retriever.search("apple", top_k=3)
        
        # doc1 should have highest score (most occurrences of "apple")
        assert results[0]['id'] == 'doc1'
        assert results[0]['score'] > results[1]['score']
        
        # Search for "banana"
        results = bm25_retriever.search("banana", top_k=3)
        
        # doc3 should have highest score (most occurrences of "banana")
        assert results[0]['id'] == 'doc3'
    
    def test_complex_filters(self, bm25_retriever, sample_documents):
        """Test complex filter operators."""
        # Add documents with numeric metadata
        docs = [
            {'id': 'doc1', 'content': 'test', 'metadata': {'score': 10, 'tags': ['a', 'b']}},
            {'id': 'doc2', 'content': 'test', 'metadata': {'score': 20, 'tags': ['b', 'c']}},
            {'id': 'doc3', 'content': 'test', 'metadata': {'score': 15, 'tags': ['a', 'c']}}
        ]
        bm25_retriever.add_documents(docs)
        
        # Test $gt operator
        results = bm25_retriever.search("test", filters={'score': {'$gt': 12}})
        assert len(results) == 2
        for r in results:
            assert r['metadata']['score'] > 12
        
        # Test $in operator
        results = bm25_retriever.search("test", filters={'tags': {'$in': ['a']}})
        doc_ids = [r['id'] for r in results]
        assert 'doc1' in doc_ids
        assert 'doc3' in doc_ids
        assert 'doc2' not in doc_ids