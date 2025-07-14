"""Main FastAPI application for KnowledgeCore Engine.

This is a simplified API entry point. For a full-featured API server,
see examples/api_server.py
"""

from fastapi import FastAPI

app = FastAPI(
    title="KnowledgeCore Engine API",
    description="A minimal API endpoint for KnowledgeCore Engine. For full functionality, use examples/api_server.py",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "KnowledgeCore Engine API",
        "version": "1.0.0",
        "note": "This is a minimal API. For full functionality, run: python examples/api_server.py"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}