"""Vector store implementation with provider abstraction."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import uuid

from ..config import RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
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
    """Result from vector search."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class VectorStore:
    """Vector store with support for multiple providers."""
    
    def __init__(self, config: RAGConfig):
        """Initialize vector store with configuration.
        
        Args:
            config: RAG configuration object
        """
        self.config = config
        self._provider = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the vector store provider."""
        if self._initialized:
            return
        
        # Create provider based on config
        if self.config.vectordb_provider == "chromadb":
            self._provider = ChromaDBProvider(self.config)
        elif self.config.vectordb_provider == "pinecone":
            self._provider = PineconeProvider(self.config)
        else:
            raise ValueError(f"Unknown vector DB provider: {self.config.vectordb_provider}")
        
        await self._provider.initialize()
        
        # Create collection
        await self._provider.create_collection(
            name=self.config.collection_name,
            dimension=self.config.embedding_dimensions
        )
        
        self._initialized = True
        
        logger.info(f"Initialized {self.config.vectordb_provider} vector store")
    
    async def add_document(self, doc: VectorDocument):
        """Add a single document to the store.
        
        Args:
            doc: Document to add
        """
        await self.initialize()
        await self._provider.add_documents([doc])
    
    async def add_documents(self, docs: List[VectorDocument]):
        """Add multiple documents to the store.
        
        Args:
            docs: List of documents to add
        """
        await self.initialize()
        
        # Process in batches if needed
        batch_size = self.config.extra_params.get("vectordb_batch_size", 100)
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            await self._provider.add_documents(batch)
            logger.debug(f"Added batch of {len(batch)} documents")
    
    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of query results ordered by relevance
        """
        await self.initialize()
        
        results = await self._provider.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=filter
        )
        
        return results
    
    async def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        await self.initialize()
        return await self._provider.get_document(doc_id)
    
    async def update_document(self, doc: VectorDocument):
        """Update an existing document.
        
        Args:
            doc: Document with updated content
        """
        await self.initialize()
        await self._provider.update_documents([doc])
    
    async def delete_document(self, doc_id: str):
        """Delete a document by ID.
        
        Args:
            doc_id: Document ID to delete
        """
        await self.initialize()
        await self._provider.delete_documents([doc_id])
    
    async def delete_documents(self, doc_ids: List[str]):
        """Delete multiple documents by IDs.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        await self.initialize()
        await self._provider.delete_documents(doc_ids)
    
    async def clear(self):
        """Clear all documents from the collection."""
        await self.initialize()
        await self._provider.clear_collection()
    
    async def list_documents(
        self,
        filter: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 20,
        return_stats: bool = True
    ) -> Dict[str, Any]:
        """列出存储中的文档。
        
        Args:
            filter: 过滤条件
            page: 页码
            page_size: 每页大小
            return_stats: 是否返回统计信息
            
        Returns:
            文档列表和分页信息
        """
        await self.initialize()
        return await self._provider.list_documents(filter, page, page_size, return_stats)
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Collection metadata
        """
        await self.initialize()
        return await self._provider.get_collection_info()


# Provider implementations

class VectorStoreProvider:
    """Base class for vector store providers."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    async def initialize(self):
        """Initialize the provider."""
        pass
    
    async def create_collection(self, name: str, dimension: int):
        """Create or get a collection."""
        raise NotImplementedError
    
    async def add_documents(self, docs: List[VectorDocument]):
        """Add documents to the collection."""
        raise NotImplementedError
    
    async def query(
        self,
        query_embedding: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Query for similar documents."""
        raise NotImplementedError
    
    async def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        raise NotImplementedError
    
    async def update_documents(self, docs: List[VectorDocument]):
        """Update existing documents."""
        raise NotImplementedError
    
    async def delete_documents(self, doc_ids: List[str]):
        """Delete documents by IDs."""
        raise NotImplementedError
    
    async def clear_collection(self):
        """Clear all documents."""
        raise NotImplementedError
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        raise NotImplementedError
    
    async def list_documents(
        self,
        filter: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 20,
        return_stats: bool = True
    ) -> Dict[str, Any]:
        """List documents in the collection."""
        raise NotImplementedError


class ChromaDBProvider(VectorStoreProvider):
    """ChromaDB vector store provider."""
    
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self._client = None
        self._collection = None
    
    async def initialize(self):
        """Initialize ChromaDB client."""
        import chromadb
        
        self._client = chromadb.PersistentClient(
            path=self.config.persist_directory
        )
        
        logger.info(f"ChromaDB initialized at {self.config.persist_directory}")
    
    async def create_collection(self, name: str, dimension: int):
        """Create or get ChromaDB collection."""
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={
                "dimension": dimension,
                "created_by": "knowledge_core_engine"
            }
        )
        
        logger.info(f"Using ChromaDB collection: {name}")
    
    async def add_documents(self, docs: List[VectorDocument]):
        """Add documents to ChromaDB."""
        if not self._collection:
            raise ValueError("Collection not initialized")
        
        ids = [doc.id for doc in docs]
        embeddings = [doc.embedding for doc in docs]
        documents = [doc.text for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    async def query(
        self,
        query_embedding: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Query ChromaDB."""
        if not self._collection:
            raise ValueError("Collection not initialized")
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        # Convert to QueryResult format
        query_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # 改进的相似度计算：使用反比例函数而不是简单的1-distance
                # 这对L2距离更合适，因为L2距离没有固定上界
                distance = results["distances"][0][i]
                similarity_score = 1.0 / (1.0 + distance)  # 将距离转换为相似度，范围(0,1]
                
                query_results.append(QueryResult(
                    id=results["ids"][0][i],
                    score=similarity_score,
                    text=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    embedding=results["embeddings"][0][i] if results.get("embeddings") else None
                ))
        
        return query_results
    
    async def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID from ChromaDB."""
        if not self._collection:
            raise ValueError("Collection not initialized")
        
        results = self._collection.get(ids=[doc_id])
        
        if results["ids"]:
            return VectorDocument(
                id=results["ids"][0],
                embedding=results["embeddings"][0] if results.get("embeddings") else [],
                text=results["documents"][0] if results["documents"] else "",
                metadata=results["metadatas"][0] if results["metadatas"] else {}
            )
        
        return None
    
    async def update_documents(self, docs: List[VectorDocument]):
        """Update documents in ChromaDB."""
        if not self._collection:
            raise ValueError("Collection not initialized")
        
        ids = [doc.id for doc in docs]
        embeddings = [doc.embedding for doc in docs]
        documents = [doc.text for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        self._collection.update(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    async def delete_documents(self, doc_ids: List[str]):
        """Delete documents from ChromaDB."""
        if not self._collection:
            raise ValueError("Collection not initialized")
        
        self._collection.delete(ids=doc_ids)
    
    async def clear_collection(self):
        """Clear all documents from collection."""
        if not self._collection:
            raise ValueError("Collection not initialized")
        
        # Get all document IDs
        all_docs = self._collection.get()
        if all_docs["ids"]:
            self._collection.delete(ids=all_docs["ids"])
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get ChromaDB collection info."""
        if not self._collection:
            raise ValueError("Collection not initialized")
        
        return {
            "name": self._collection.name,
            "count": self._collection.count(),
            "metadata": self._collection.metadata
        }
    
    async def list_documents(
        self,
        filter: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 20,
        return_stats: bool = True
    ) -> Dict[str, Any]:
        """List documents in ChromaDB collection with pagination."""
        if not self._collection:
            raise ValueError("Collection not initialized")
        
        # 获取所有文档（ChromaDB不直接支持分页，需要手动实现）
        all_results = self._collection.get(where=filter)
        
        if not all_results["ids"]:
            return {
                "documents": [],
                "total": 0,
                "page": page,
                "page_size": page_size,
                "pages": 0
            }
        
        # 按文档名分组（从doc_id中提取）
        doc_groups = {}
        for i, doc_id in enumerate(all_results["ids"]):
            # 从ID中提取文档名（格式：filename_chunkindex_startchar）
            parts = doc_id.rsplit("_", 2)
            if len(parts) >= 3:
                doc_name = parts[0]
            else:
                doc_name = doc_id  # 如果格式不符，使用完整ID
            
            if doc_name not in doc_groups:
                doc_groups[doc_name] = {
                    "name": doc_name,
                    "chunks": [],
                    "metadata": all_results["metadatas"][i] if all_results["metadatas"] else {}
                }
            
            # 收集chunk信息
            doc_groups[doc_name]["chunks"].append({
                "id": doc_id,
                "size": len(all_results["documents"][i]) if all_results["documents"] else 0
            })
        
        # 构建文档列表
        documents = []
        for doc_name, doc_info in doc_groups.items():
            doc_entry = {
                "name": doc_info["metadata"].get("source", doc_name),
                "path": doc_info["metadata"].get("file_path", ""),
                "metadata": doc_info["metadata"]
            }
            
            if return_stats:
                doc_entry["chunks_count"] = len(doc_info["chunks"])
                doc_entry["total_size"] = sum(chunk["size"] for chunk in doc_info["chunks"])
            
            # 添加创建时间（如果有）
            if "created_at" in doc_info["metadata"]:
                doc_entry["created_at"] = doc_info["metadata"]["created_at"]
            
            documents.append(doc_entry)
        
        # 按名称排序
        documents.sort(key=lambda x: x["name"])
        
        # 实现分页
        total_docs = len(documents)
        total_pages = (total_docs + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = documents[start_idx:end_idx]
        
        return {
            "documents": paginated_docs,
            "total": total_docs,
            "page": page,
            "page_size": page_size,
            "pages": total_pages
        }


class PineconeProvider(VectorStoreProvider):
    """Pinecone vector store provider (placeholder)."""
    
    async def initialize(self):
        """Initialize Pinecone client."""
        if not self.config.vectordb_api_key:
            raise ValueError("Pinecone API key is required")
        
        # In real implementation:
        # import pinecone
        # pinecone.init(api_key=self.config.vectordb_api_key)
        
        logger.info("Pinecone provider initialized")
    
    async def create_collection(self, name: str, dimension: int):
        """Create or get Pinecone index."""
        # Placeholder implementation
        logger.info(f"Using Pinecone index: {name}")
    
    async def add_documents(self, docs: List[VectorDocument]):
        """Add documents to Pinecone."""
        # Placeholder
        pass
    
    async def query(
        self,
        query_embedding: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Query Pinecone."""
        # Placeholder
        return []
    
    async def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document from Pinecone."""
        # Placeholder
        return None
    
    async def update_documents(self, docs: List[VectorDocument]):
        """Update documents in Pinecone."""
        # Placeholder
        pass
    
    async def delete_documents(self, doc_ids: List[str]):
        """Delete from Pinecone."""
        # Placeholder
        pass
    
    async def clear_collection(self):
        """Clear Pinecone index."""
        # Placeholder
        pass
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get Pinecone index info."""
        # Placeholder
        return {"name": self.config.collection_name, "provider": "pinecone"}