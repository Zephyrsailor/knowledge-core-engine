"""
Simple API server example for KnowledgeCore Engine.

This example demonstrates how to expose the RAG system as a REST API
using the existing simple_usage pattern.
"""

import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.core.parsing.document_processor import DocumentProcessor
from knowledge_core_engine.core.chunking.pipeline import ChunkingPipeline
from knowledge_core_engine.core.embedding.embedder import TextEmbedder
from knowledge_core_engine.core.embedding.vector_store import VectorStore
from knowledge_core_engine.core.retrieval.retriever import Retriever
from knowledge_core_engine.core.generation.generator import Generator


# API Models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    top_k: int = 5
    include_citations: bool = True


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    answer: str
    citations: List[dict]
    timestamp: str


class DocumentInfo(BaseModel):
    """Document information model."""
    document_id: str
    title: str
    chunks: int
    created_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: dict


# Global instances
config: Optional[RAGConfig] = None
document_processor: Optional[DocumentProcessor] = None
chunking_pipeline: Optional[ChunkingPipeline] = None
embedder: Optional[TextEmbedder] = None
vector_store: Optional[VectorStore] = None
retriever: Optional[Retriever] = None
generator: Optional[Generator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global config, document_processor, chunking_pipeline
    global embedder, vector_store, retriever, generator
    
    print("🚀 Initializing KnowledgeCore Engine API...")
    
    # Create configuration
    config = RAGConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
        llm_api_key=os.getenv("DEEPSEEK_API_KEY"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "dashscope"),
        embedding_api_key=os.getenv("DASHSCOPE_API_KEY"),
        vectordb_provider="chromadb",
        persist_directory="./data/api_chroma_db",
        include_citations=True
    )
    
    # Initialize components
    document_processor = DocumentProcessor()
    chunking_pipeline = ChunkingPipeline(enable_smart_chunking=True)
    embedder = TextEmbedder(config)
    vector_store = VectorStore(config)
    retriever = Retriever(config)
    generator = Generator(config)
    
    # Initialize async components
    await embedder.initialize()
    await vector_store.initialize()
    await retriever.initialize()
    await generator.initialize()
    
    print("✅ API server ready!")
    
    yield
    
    # Cleanup
    print("👋 API server shutting down")


# Initialize FastAPI app
app = FastAPI(
    title="KnowledgeCore Engine API",
    description="RESTful API for RAG-based knowledge management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "name": "KnowledgeCore Engine API",
        "version": "1.0.0",
        "description": "RAG-based knowledge management system"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "document_processor": document_processor is not None,
            "chunking": chunking_pipeline is not None,
            "embedder": embedder is not None,
            "vector_store": vector_store is not None,
            "retriever": retriever is not None,
            "generator": generator is not None
        }
    )


@app.post("/documents/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    if not all([document_processor, chunking_pipeline, embedder, vector_store]):
        raise HTTPException(status_code=503, detail="Components not initialized")
    
    # Save uploaded file
    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process document
    try:
        # Parse
        parse_result = await document_processor.process(str(file_path))
        
        # Chunk
        chunk_result = await chunking_pipeline.process_parse_result(parse_result)
        
        # Embed and store
        chunks_stored = 0
        for chunk in chunk_result.chunks:
            embedding_result = await embedder.embed(chunk.content)
            await vector_store.add_documents([{
                "id": chunk.chunk_id,
                "content": chunk.content,
                "embedding": embedding_result.embedding,
                "metadata": chunk.metadata
            }])
            chunks_stored += 1
        
        return DocumentInfo(
            document_id=f"doc_{hash(file.filename)}",
            title=file.filename,
            chunks=chunks_stored,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base."""
    if not all([retriever, generator]):
        raise HTTPException(status_code=503, detail="Components not initialized")
    
    try:
        # Retrieve relevant contexts
        contexts = await retriever.retrieve(request.query, top_k=request.top_k)
        
        if not contexts:
            return QueryResponse(
                query=request.query,
                answer="抱歉，我在知识库中没有找到相关信息。请尝试其他问题或上传更多文档。",
                citations=[],
                timestamp=datetime.now().isoformat()
            )
        
        # Generate answer
        answer = await generator.generate(request.query, contexts)
        
        # Format citations
        citations = []
        if answer.citations:
            for citation in answer.citations:
                citations.append({
                    "index": citation.index,
                    "source": citation.document_title,
                    "text": citation.text[:200] + "..." if len(citation.text) > 200 else citation.text
                })
        
        return QueryResponse(
            query=request.query,
            answer=answer.answer,
            citations=citations,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "api_server_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )