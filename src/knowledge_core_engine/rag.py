"""Simple RAG implementation - one good way to do things."""

from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass

from .config import Config


@dataclass
class Document:
    """A document to add to knowledge base."""
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass  
class SearchResult:
    """A search result."""
    content: str
    score: float
    metadata: Dict[str, Any]


class RAG:
    """Simple RAG system - retrieve and generate."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize RAG system.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self._initialized = False
        
        # These will be initialized in setup()
        self._llm = None
        self._embedder = None
        self._vectordb = None
    
    async def setup(self):
        """Set up the RAG system."""
        if self._initialized:
            return
        
        # Initialize LLM
        if self.config.llm_provider == "deepseek":
            from .providers.llm import DeepSeekProvider
            self._llm = DeepSeekProvider(self.config)
        elif self.config.llm_provider == "qwen":
            from .providers.llm import QwenProvider
            self._llm = QwenProvider(self.config)
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")
        
        # Initialize Embedder
        if self.config.embedding_provider == "dashscope":
            from .providers.embedding import DashScopeEmbedder
            self._embedder = DashScopeEmbedder(self.config)
        elif self.config.embedding_provider == "openai":
            from .providers.embedding import OpenAIEmbedder
            self._embedder = OpenAIEmbedder(self.config)
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embedding_provider}")
        
        # Initialize Vector DB
        if self.config.vectordb_provider == "chromadb":
            from .providers.vectordb import ChromaDB
            self._vectordb = ChromaDB(self.config)
        else:
            raise ValueError(f"Unknown vector DB provider: {self.config.vectordb_provider}")
        
        # Initialize all components
        await self._llm.setup()
        await self._embedder.setup()
        await self._vectordb.setup(dimension=self.config.embedding_dim)
        
        self._initialized = True
    
    async def add(self, documents: List[Document]):
        """Add documents to knowledge base.
        
        Args:
            documents: List of documents to add
        """
        await self.setup()
        
        for doc in documents:
            # Prepare text for embedding
            if self.config.use_multi_vector and doc.metadata:
                text = self._prepare_multi_vector(doc.content, doc.metadata)
            else:
                text = doc.content
            
            # Get embedding
            embedding = await self._embedder.embed(text)
            
            # Store in vector DB
            await self._vectordb.add(
                id=doc.metadata.get("id", str(hash(doc.content))),
                embedding=embedding,
                content=doc.content,
                metadata=doc.metadata
            )
    
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        await self.setup()
        
        # Get query embedding
        query_embedding = await self._embedder.embed(query)
        
        # Search in vector DB
        results = await self._vectordb.search(query_embedding, top_k)
        
        return [
            SearchResult(
                content=r["content"],
                score=r["score"],
                metadata=r["metadata"]
            )
            for r in results
        ]
    
    async def ask(self, question: str, top_k: int = 5) -> str:
        """Ask a question and get an answer.
        
        Args:
            question: The question to ask
            top_k: Number of documents to retrieve
            
        Returns:
            Generated answer
        """
        await self.setup()
        
        # Search for relevant documents
        docs = await self.search(question, top_k)
        
        # Build prompt
        prompt = self._build_prompt(question, docs)
        
        # Generate answer
        answer = await self._llm.complete(prompt)
        
        return answer
    
    def _prepare_multi_vector(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare text using multi-vector strategy."""
        parts = [f"Content: {content}"]
        
        if metadata.get("summary"):
            parts.append(f"Summary: {metadata['summary']}")
        
        if metadata.get("questions"):
            questions = metadata["questions"]
            if isinstance(questions, list):
                parts.append(f"Questions: {' '.join(questions)}")
        
        return "\n\n".join(parts)
    
    def _build_prompt(self, question: str, docs: List[SearchResult]) -> str:
        """Build prompt for LLM."""
        prompt = "基于以下文档回答用户问题。\n\n"
        
        for i, doc in enumerate(docs, 1):
            prompt += f"文档{i} (相关度: {doc.score:.2f}):\n"
            prompt += f"{doc.content}\n\n"
        
        prompt += f"问题: {question}\n"
        prompt += "回答: "
        
        return prompt


# Simple provider implementations (placeholders)

class SimpleProvider:
    """Base class for simple providers."""
    def __init__(self, config: Config):
        self.config = config
    
    async def setup(self):
        """Setup the provider."""
        pass


class DeepSeekProvider(SimpleProvider):
    """Simple DeepSeek LLM provider."""
    async def complete(self, prompt: str) -> str:
        # Actual implementation would call DeepSeek API
        return "DeepSeek response placeholder"


class DashScopeEmbedder(SimpleProvider):
    """Simple DashScope embedding provider."""
    async def embed(self, text: str) -> List[float]:
        # Actual implementation would call DashScope API
        return [0.1] * self.config.embedding_dim


class ChromaDB(SimpleProvider):
    """Simple ChromaDB vector store."""
    def __init__(self, config: Config):
        super().__init__(config)
        self._collection = None
    
    async def setup(self, dimension: int):
        """Setup ChromaDB."""
        import chromadb
        client = chromadb.PersistentClient(path=self.config.vectordb_persist_dir)
        self._collection = client.get_or_create_collection(
            name=self.config.vectordb_collection
        )
    
    async def add(self, id: str, embedding: List[float], content: str, metadata: Dict):
        """Add document to ChromaDB."""
        self._collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
    
    async def search(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Search in ChromaDB."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        return output