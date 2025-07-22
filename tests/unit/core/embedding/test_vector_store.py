"""Unit tests for vector store module."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List, Dict, Any

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.embedding.vector_store import (
    VectorStore, VectorDocument, QueryResult, ChromaDBProvider
)
from knowledge_core_engine.core.embedding.embedder import EmbeddingResult


class TestVectorDocument:
    """Test VectorDocument class."""
    
    def test_vector_document_creation(self):
        """Test creating a VectorDocument."""
        doc = VectorDocument(
            id="test_id",
            embedding=[0.1, 0.2, 0.3],
            text="Test text",
            metadata={"chunk_id": "chunk_1", "score": 0.95}
        )
        
        assert doc.id == "test_id"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.text == "Test text"
        assert doc.metadata["chunk_id"] == "chunk_1"
        assert doc.metadata["score"] == 0.95
    
    def test_vector_document_auto_id(self):
        """Test VectorDocument with auto-generated ID."""
        doc = VectorDocument(
            id="",
            embedding=[0.1, 0.2],
            text="Test"
        )
        
        # Should have generated an ID
        assert doc.id is not None
        assert len(doc.id) > 0


class TestQueryResult:
    """Test QueryResult class."""
    
    def test_query_result_creation(self):
        """Test creating a QueryResult."""
        result = QueryResult(
            id="result_1",
            score=0.95,
            text="Result text",
            metadata={"source": "test"},
            embedding=[0.1, 0.2]
        )
        
        assert result.id == "result_1"
        assert result.score == 0.95
        assert result.text == "Result text"
        assert result.metadata["source"] == "test"
        assert result.embedding == [0.1, 0.2]


class TestVectorStore:
    """Test VectorStore class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGConfig(
            vectordb_provider="chromadb",
            collection_name="test_collection",
            persist_directory="./test_data"
        )
    
    @pytest.fixture
    def vector_store(self, config):
        """Create VectorStore instance."""
        return VectorStore(config)
    
    @pytest.mark.asyncio
    async def test_vector_store_initialization(self, vector_store):
        """Test VectorStore initialization."""
        assert vector_store._initialized is False
        assert vector_store.config.vectordb_provider == "chromadb"
        assert vector_store.config.collection_name == "test_collection"
    
    @pytest.mark.asyncio
    async def test_initialize_chromadb(self, vector_store):
        """Test initializing ChromaDB provider."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            await vector_store.initialize()
            
            assert vector_store._initialized is True
            assert vector_store._provider is not None
            assert isinstance(vector_store._provider, ChromaDBProvider)
    
    @pytest.mark.asyncio
    async def test_add_single_document(self, vector_store):
        """Test adding a single document."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            doc = VectorDocument(
                id="doc_1",
                embedding=[0.1, 0.2, 0.3],
                text="Test document",
                metadata={"type": "test"}
            )
            
            await vector_store.add_document(doc)
            
            # Verify collection.add was called
            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args[1]
            assert call_args["ids"] == ["doc_1"]
            assert call_args["embeddings"] == [[0.1, 0.2, 0.3]]
            assert call_args["documents"] == ["Test document"]
            assert call_args["metadatas"] == [{"type": "test"}]
    
    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, vector_store):
        """Test adding multiple documents."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            docs = [
                VectorDocument(
                    id=f"doc_{i}",
                    embedding=[0.1 * i] * 3,
                    text=f"Document {i}",
                    metadata={"index": i}
                )
                for i in range(3)
            ]
            
            await vector_store.add_documents(docs)
            
            # Should batch add documents
            assert mock_collection.add.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_query_vectors(self, vector_store):
        """Test querying by vector similarity."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Mock query response
            mock_collection.query.return_value = {
                "ids": [["doc_1", "doc_2"]],
                "distances": [[0.1, 0.2]],
                "documents": [["Document 1", "Document 2"]],
                "metadatas": [[{"score": 0.9}, {"score": 0.8}]]
            }
            
            query_embedding = [0.1, 0.2, 0.3]
            results = await vector_store.query(
                query_embedding=query_embedding,
                top_k=2
            )
            
            assert len(results) == 2
            assert results[0].id == "doc_1"
            # Score计算: 1.0 / (1.0 + distance) = 1.0 / (1.0 + 0.1) = 1.0 / 1.1 ≈ 0.909
            assert abs(results[0].score - (1.0 / 1.1)) < 0.001  # 使用近似比较避免浮点精度问题
            assert results[0].text == "Document 1"
    
    @pytest.mark.asyncio
    async def test_get_document_by_id(self, vector_store):
        """Test retrieving document by ID."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            mock_collection.get.return_value = {
                "ids": ["doc_1"],
                "embeddings": [[0.1, 0.2, 0.3]],
                "documents": ["Document 1"],
                "metadatas": [{"type": "test"}]
            }
            
            doc = await vector_store.get_document("doc_1")
            
            assert doc is not None
            assert doc.id == "doc_1"
            assert doc.text == "Document 1"
            assert doc.embedding == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_update_document(self, vector_store):
        """Test updating an existing document."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            doc = VectorDocument(
                id="existing_doc",
                embedding=[0.5, 0.5],
                text="Updated text",
                metadata={"version": 2}
            )
            
            await vector_store.update_document(doc)
            
            mock_collection.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_document(self, vector_store):
        """Test deleting a document."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            await vector_store.delete_document("doc_to_delete")
            
            mock_collection.delete.assert_called_once_with(ids=["doc_to_delete"])
    
    @pytest.mark.asyncio
    async def test_clear_collection(self, vector_store):
        """Test clearing all documents."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Mock getting all IDs
            mock_collection.get.return_value = {
                "ids": ["doc_1", "doc_2", "doc_3"]
            }
            
            await vector_store.clear()
            
            # Should delete all documents
            mock_collection.delete.assert_called_once_with(
                ids=["doc_1", "doc_2", "doc_3"]
            )
    
    @pytest.mark.asyncio
    async def test_get_collection_info(self, vector_store):
        """Test getting collection information."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_collection.name = "test_collection"
            mock_collection.count.return_value = 100
            mock_collection.metadata = {"created_by": "test"}
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            info = await vector_store.get_collection_info()
            
            assert info["name"] == "test_collection"
            assert info["count"] == 100
            assert info["metadata"]["created_by"] == "test"
    
    @pytest.mark.asyncio 
    async def test_batch_processing(self, vector_store):
        """Test batch processing of documents."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Set small batch size
            vector_store.config.extra_params["vectordb_batch_size"] = 2
            
            # Create 5 documents
            docs = [
                VectorDocument(
                    id=f"doc_{i}",
                    embedding=[0.1] * 10,
                    text=f"Document {i}",
                    metadata={}
                )
                for i in range(5)
            ]
            
            await vector_store.add_documents(docs)
            
            # Should be called 3 times (2 + 2 + 1)
            assert mock_collection.add.call_count == 3


class TestChromaDBProvider:
    """Test ChromaDB provider specifically."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGConfig(
            vectordb_provider="chromadb",
            collection_name="test_chromadb"
        )
    
    @pytest.fixture
    def provider(self, config):
        """Create ChromaDB provider."""
        return ChromaDBProvider(config)
    
    @pytest.mark.asyncio
    async def test_provider_initialization(self, provider):
        """Test ChromaDB provider initialization."""
        with patch('chromadb.PersistentClient') as mock_client:
            await provider.initialize()
            
            mock_client.assert_called_once_with(
                path=provider.config.persist_directory
            )
    
    @pytest.mark.asyncio
    async def test_create_collection(self, provider):
        """Test creating ChromaDB collection."""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            await provider.initialize()
            await provider.create_collection("test_collection", 1536)
            
            mock_client.return_value.get_or_create_collection.assert_called_once()
            call_args = mock_client.return_value.get_or_create_collection.call_args[1]
            assert call_args["name"] == "test_collection"
            assert call_args["metadata"]["dimension"] == 1536