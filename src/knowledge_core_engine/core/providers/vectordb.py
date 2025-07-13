"""Vector database provider abstraction for flexible storage."""

from abc import abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from .base import Provider, ProviderConfig, ProviderFactory


@dataclass
class VectorDBConfig(ProviderConfig):
    """VectorDB-specific configuration."""
    collection_name: str = "knowledge_core"
    distance_metric: str = "cosine"  # cosine, l2, ip
    persist_directory: Optional[str] = None
    index_params: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.index_params is None:
            self.index_params = {}


@dataclass
class Document:
    """Document to store in vector database."""
    id: str
    embedding: List[float]
    text: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class QueryResult:
    """Query result from vector database."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class VectorDBProvider(Provider):
    """Base class for vector database providers."""
    
    @abstractmethod
    async def create_collection(self, dimension: int, **kwargs):
        """Create or get collection.
        
        Args:
            dimension: Vector dimension
            **kwargs: Provider-specific parameters
        """
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]):
        """Add documents to the collection.
        
        Args:
            documents: List of documents to add
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter: Metadata filter
            
        Returns:
            List of query results
        """
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]):
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    async def update(self, documents: List[Document]):
        """Update existing documents."""
        pass
    
    async def get(self, id: str) -> Optional[Document]:
        """Get document by ID."""
        results = await self.search_by_ids([id])
        return results[0] if results else None
    
    async def search_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs (default implementation)."""
        # Providers can override for more efficient implementation
        raise NotImplementedError("search_by_ids not implemented")


# --- ChromaDB Implementation ---

class ChromaDBProvider(VectorDBProvider):
    """ChromaDB vector database provider."""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self._client = None
        self._collection = None
    
    async def initialize(self):
        """Initialize ChromaDB client."""
        import chromadb
        
        if self.config.persist_directory:
            self._client = chromadb.PersistentClient(
                path=self.config.persist_directory
            )
        else:
            self._client = chromadb.Client()
    
    async def create_collection(self, dimension: int, **kwargs):
        """Create or get ChromaDB collection."""
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"dimension": dimension, **kwargs}
        )
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to ChromaDB."""
        if not self._collection:
            raise ValueError("Collection not created. Call create_collection first.")
        
        self._collection.add(
            ids=[doc.id for doc in documents],
            embeddings=[doc.embedding for doc in documents],
            documents=[doc.text for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Search in ChromaDB."""
        if not self._collection:
            raise ValueError("Collection not created.")
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        # Convert to QueryResult format
        query_results = []
        for i in range(len(results["ids"][0])):
            query_results.append(QueryResult(
                id=results["ids"][0][i],
                score=1 - results["distances"][0][i],  # Convert distance to similarity
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                embedding=results["embeddings"][0][i] if results.get("embeddings") else None
            ))
        
        return query_results
    
    async def delete(self, ids: List[str]):
        """Delete documents from ChromaDB."""
        if self._collection:
            self._collection.delete(ids=ids)
    
    async def update(self, documents: List[Document]):
        """Update documents in ChromaDB."""
        if not self._collection:
            raise ValueError("Collection not created.")
        
        self._collection.update(
            ids=[doc.id for doc in documents],
            embeddings=[doc.embedding for doc in documents],
            documents=[doc.text for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
    
    async def close(self):
        """Clean up resources."""
        self._collection = None
        self._client = None


# --- Pinecone Implementation (Placeholder) ---

class PineconeProvider(VectorDBProvider):
    """Pinecone vector database provider."""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self._index = None
    
    async def initialize(self):
        """Initialize Pinecone client."""
        if not self.config.api_key:
            raise ValueError("Pinecone API key is required")
        # import pinecone
        # pinecone.init(api_key=self.config.api_key)
    
    async def create_collection(self, dimension: int, **kwargs):
        """Create or get Pinecone index."""
        # self._index = pinecone.Index(self.config.collection_name)
        pass
    
    async def add_documents(self, documents: List[Document]):
        """Add documents to Pinecone."""
        # Implementation would batch upsert to Pinecone
        pass
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Search in Pinecone."""
        # Implementation would query Pinecone
        return []
    
    async def delete(self, ids: List[str]):
        """Delete from Pinecone."""
        pass
    
    async def update(self, documents: List[Document]):
        """Update in Pinecone."""
        # Pinecone uses upsert for updates
        await self.add_documents(documents)
    
    async def close(self):
        """Clean up resources."""
        self._index = None


# Register providers
ProviderFactory.register("vectordb", "chromadb", ChromaDBProvider)
ProviderFactory.register("vectordb", "pinecone", PineconeProvider)


# --- Usage Example ---
"""
# Configuration
config = {
    "provider": "chromadb",
    "collection_name": "my_knowledge_base",
    "persist_directory": "./chroma_data",
    "distance_metric": "cosine"
}

# Create vector DB provider
vectordb = ProviderFactory.create("vectordb", config)
await vectordb.initialize()
await vectordb.create_collection(dimension=1536)

# Add documents
docs = [
    Document(
        id="doc1",
        embedding=[0.1] * 1536,
        text="RAG combines retrieval and generation",
        metadata={"source": "paper1", "chunk_type": "definition"}
    )
]
await vectordb.add_documents(docs)

# Search
results = await vectordb.search(
    query_embedding=[0.15] * 1536,
    top_k=5,
    filter={"chunk_type": "definition"}
)

for result in results:
    print(f"Score: {result.score:.3f}, Text: {result.text[:50]}...")
"""