"""Elasticsearch-based BM25 implementation."""

import logging
from typing import List, Dict, Any, Optional
import hashlib

from .base import BaseBM25Retriever, BM25Result

logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseBM25Retriever):
    """BM25 retriever using Elasticsearch.
    
    This is for users who need enterprise-grade features like:
    - Distributed search
    - Real-time indexing
    - Advanced query DSL
    - Aggregations
    """
    
    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "knowledge_core",
        k1: float = 1.2,
        b: float = 0.75,
        **kwargs
    ):
        """Initialize Elasticsearch retriever.
        
        Args:
            es_url: Elasticsearch URL
            index_name: Name of the index
            k1: BM25 k1 parameter
            b: BM25 b parameter
            **kwargs: Additional Elasticsearch client options
        """
        super().__init__()
        self.es_url = es_url
        self.index_name = index_name
        self.k1 = k1
        self.b = b
        self.es_options = kwargs
        self._client = None
    
    async def _initialize(self) -> None:
        """Initialize Elasticsearch client."""
        try:
            from elasticsearch import AsyncElasticsearch
            
            self._client = AsyncElasticsearch(
                self.es_url,
                **self.es_options
            )
            
            # Check connection
            info = await self._client.info()
            logger.info(f"Connected to Elasticsearch {info['version']['number']}")
            
            # Create index if it doesn't exist
            await self._ensure_index()
            
        except ImportError:
            raise RuntimeError(
                "Elasticsearch not installed. Please install with: "
                "pip install 'knowledge-core-engine[elasticsearch]'"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Elasticsearch: {e}")
    
    async def _ensure_index(self) -> None:
        """Ensure the index exists with proper settings."""
        exists = await self._client.indices.exists(index=self.index_name)
        
        if not exists:
            # Create index with BM25 similarity
            settings = {
                "settings": {
                    "index": {
                        "similarity": {
                            "default": {
                                "type": "BM25",
                                "k1": self.k1,
                                "b": self.b
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"  # Can be customized
                        },
                        "doc_id": {
                            "type": "keyword"
                        },
                        "metadata": {
                            "type": "object",
                            "enabled": True
                        }
                    }
                }
            }
            
            await self._client.indices.create(
                index=self.index_name,
                body=settings
            )
            logger.info(f"Created index: {self.index_name}")
    
    async def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to Elasticsearch."""
        if not self._initialized:
            await self.initialize()
        
        if not documents:
            return
        
        # Generate doc IDs if not provided
        if doc_ids is None:
            doc_ids = [self._generate_es_id(doc) for doc in documents]
        
        # Ensure metadata list matches documents
        if metadata is None:
            metadata = [{} for _ in documents]
        
        # Bulk index documents
        bulk_body = []
        for doc_id, content, meta in zip(doc_ids, documents, metadata):
            # Index action
            bulk_body.append({
                "index": {
                    "_index": self.index_name,
                    "_id": doc_id
                }
            })
            # Document
            bulk_body.append({
                "doc_id": doc_id,
                "content": content,
                "metadata": meta
            })
        
        # Perform bulk indexing
        response = await self._client.bulk(body=bulk_body, refresh=True)
        
        if response["errors"]:
            logger.error("Some documents failed to index")
            for item in response["items"]:
                if "error" in item["index"]:
                    logger.error(f"Error: {item['index']['error']}")
        else:
            logger.info(f"Indexed {len(documents)} documents")
        
        # Store for compatibility
        self._documents.extend(documents)
        self._doc_ids.extend(doc_ids)
        self._metadata.extend(metadata)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[BM25Result]:
        """Search using Elasticsearch BM25."""
        if not self._initialized:
            await self.initialize()
        
        # Build query
        es_query = {
            "match": {
                "content": {
                    "query": query,
                    "operator": "or"
                }
            }
        }
        
        # Add metadata filters if provided
        if filter_metadata:
            filters = []
            for key, value in filter_metadata.items():
                filters.append({
                    "term": {f"metadata.{key}": value}
                })
            
            es_query = {
                "bool": {
                    "must": es_query,
                    "filter": filters
                }
            }
        
        # Execute search
        response = await self._client.search(
            index=self.index_name,
            body={
                "query": es_query,
                "size": top_k,
                "_source": ["doc_id", "content", "metadata"]
            }
        )
        
        # Parse results
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(BM25Result(
                document_id=source["doc_id"],
                document=source["content"],
                score=hit["_score"],
                metadata=source.get("metadata", {})
            ))
        
        return results
    
    async def clear(self) -> None:
        """Clear all documents from the index."""
        if not self._initialized:
            await self.initialize()
        
        # Delete all documents
        await self._client.delete_by_query(
            index=self.index_name,
            body={
                "query": {"match_all": {}}
            }
        )
        
        self._documents = []
        self._doc_ids = []
        self._metadata = []
        
        logger.info("Elasticsearch index cleared")
    
    def _generate_es_id(self, content: str) -> str:
        """Generate a unique ID for Elasticsearch."""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _close(self) -> None:
        """Close Elasticsearch client."""
        if self._client:
            await self._client.close()
            self._client = None