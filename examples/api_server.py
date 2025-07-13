"""
API server example for KnowledgeCore Engine.

This example demonstrates how to expose the RAG system as a REST API.
"""

import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from knowledge_core_engine.core.config import RAGConfig
from knowledge_core_engine.pipelines.ingestion import IngestionPipeline
from knowledge_core_engine.pipelines.retrieval import RetrievalPipeline
from knowledge_core_engine.pipelines.generation import GenerationPipeline


# API Models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    top_k: int = 5
    include_citations: bool = True
    stream: bool = False
    filters: Optional[dict] = None


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    answer: str
    citations: List[dict]
    usage: dict
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
    pipelines: dict


# Initialize FastAPI app
app = FastAPI(
    title="KnowledgeCore Engine API",
    description="RESTful API for RAG-based knowledge management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
config = RAGConfig(
    # LLM Configuration
    llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
    llm_model=os.getenv("LLM_MODEL", "deepseek-chat"),
    llm_api_key=os.getenv("LLM_API_KEY"),
    
    # Embedding Configuration
    embedding_provider=os.getenv("EMBEDDING_PROVIDER", "dashscope"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
    embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
    
    # Vector Store Configuration
    vector_store_provider="chromadb",
    vector_store_path="./data/api_chroma_db",
    
    # Generation Settings
    temperature=0.1,
    max_tokens=2048,
    include_citations=True,
    
    # Extra Parameters
    extra_params={
        "language": "zh",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "enable_metadata_enhancement": True,
        "citation_style": "inline"
    }
)

# Pipeline instances (initialized on startup)
ingestion_pipeline: Optional[IngestionPipeline] = None
retrieval_pipeline: Optional[RetrievalPipeline] = None
generation_pipeline: Optional[GenerationPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipelines on startup."""
    global ingestion_pipeline, retrieval_pipeline, generation_pipeline
    
    print("🚀 Initializing KnowledgeCore Engine API...")
    
    # Initialize pipelines
    ingestion_pipeline = IngestionPipeline(config)
    await ingestion_pipeline.initialize()
    
    retrieval_pipeline = RetrievalPipeline(config)
    await retrieval_pipeline.initialize()
    
    generation_pipeline = GenerationPipeline(config)
    await generation_pipeline.initialize()
    
    print("✅ API server ready!")


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
        pipelines={
            "ingestion": ingestion_pipeline is not None,
            "retrieval": retrieval_pipeline is not None,
            "generation": generation_pipeline is not None
        }
    )


@app.post("/documents/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")
    
    # Save uploaded file
    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process document
    try:
        result = await ingestion_pipeline.process_document(file_path)
        
        return DocumentInfo(
            document_id=result["document_id"],
            title=result.get("title", file.filename),
            chunks=result["chunks_created"],
            created_at=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base."""
    if not retrieval_pipeline or not generation_pipeline:
        raise HTTPException(status_code=503, detail="Pipelines not initialized")
    
    try:
        # Retrieve contexts
        contexts = await retrieval_pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        if not contexts:
            return QueryResponse(
                query=request.query,
                answer="抱歉，我在知识库中没有找到相关信息。请尝试其他问题或上传更多文档。",
                citations=[],
                usage={"total_tokens": 0},
                timestamp=datetime.now().isoformat()
            )
        
        # Generate answer
        result = await generation_pipeline.generate(
            query=request.query,
            contexts=contexts
        )
        
        # Format response
        return QueryResponse(
            query=request.query,
            answer=result.answer,
            citations=[c.to_dict() for c in result.citations],
            usage=result.usage,
            timestamp=result.timestamp.isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all documents in the knowledge base."""
    # This is a placeholder - in production, you'd query the document store
    # For now, return empty list
    return []


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    # This is a placeholder - in production, you'd implement document deletion
    return {"message": f"Document {document_id} deletion not implemented yet"}


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """Stream query response (SSE endpoint)."""
    from fastapi.responses import StreamingResponse
    import json
    
    if not retrieval_pipeline or not generation_pipeline:
        raise HTTPException(status_code=503, detail="Pipelines not initialized")
    
    async def generate():
        # Retrieve contexts
        contexts = await retrieval_pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Stream generation
        async for chunk in generation_pipeline.stream_generate(
            query=request.query,
            contexts=contexts
        ):
            data = {
                "content": chunk.content,
                "is_final": chunk.is_final,
                "citations": [c.to_dict() for c in chunk.citations] if chunk.citations else None,
                "usage": chunk.usage
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.get("/stats")
async def get_statistics():
    """Get system statistics."""
    # This is a placeholder - in production, you'd gather real statistics
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "total_queries": 0,
        "average_response_time": 0,
        "storage_used_mb": 0
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )