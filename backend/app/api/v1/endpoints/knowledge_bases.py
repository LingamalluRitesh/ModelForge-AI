"""
ModelForge AI - Knowledge Bases & RAG API Endpoints
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.services.vector_knowledge_service import VectorKnowledgeService

router = APIRouter()


class IngestDocumentRequest(BaseModel):
    document_name: str
    document_text: str
    chunk_size: int = Field(500, ge=50, le=2000)
    chunk_overlap: int = Field(50, ge=0, le=500)


class SearchKnowledgeBaseRequest(BaseModel):
    query_text: str
    top_k: int = Field(5, ge=1, le=50)
    mock_chunks: Optional[List[Dict[str, Any]]] = None


@router.post("/ingest")
async def ingest_knowledge_document(request: IngestDocumentRequest):
    """Chunk and embed document for HNSW vector retrieval."""
    try:
        chunks = VectorKnowledgeService.chunk_and_embed_document(
            document_text=request.document_text,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return {
            "document_name": request.document_name,
            "chunks_generated": len(chunks),
            "chunks": chunks,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}",
        )


@router.post("/search")
async def search_knowledge_base(request: SearchKnowledgeBaseRequest):
    """Execute low-latency dense semantic vector search."""
    try:
        chunks = request.mock_chunks or VectorKnowledgeService.chunk_and_embed_document(
            "ModelForge AI provides high-performance real-time model inference and feature store."
        )
        res = VectorKnowledgeService.semantic_search(
            query_text=request.query_text,
            knowledge_base_chunks=chunks,
            top_k=request.top_k,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}",
        )
