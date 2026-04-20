# routers/rag.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Basic RAG imports
from services.pdf_service import extract_text_from_pdf, chunk_documents
from services.vector_store import get_vector_store
from services.rag_service import query_rag, get_enhanced_rag_service

# Models
from models.schemas import (
    QueryRequest, QueryResponse, IngestResponse,
    CuttingFeaturesRequest,
    RiskExplanationResponse,
    RecommendationsResponse,
    AlertResponse,
    SourceDocument
)

# Create router
router = APIRouter(prefix="/rag", tags=["RAG"])


# ============================================
# BASIC RAG ENDPOINTS (Document Q&A)
# ============================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF document to the knowledge base"""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        contents = await file.read()
        documents = extract_text_from_pdf(contents, file.filename)

        if not documents:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        chunks = chunk_documents(documents)
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        return IngestResponse(
            message="PDF ingested successfully.",
            filename=file.filename,
            chunks_stored=len(chunks)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Pass the role to query_rag
        result = await query_rag(
            question=request.question,
            role=request.role  # ← ADD THIS LINE
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/collection")
async def clear_collection():
    try:
        from services.vector_store import clear_vector_store
        clear_vector_store()
        return {"message": "Collection cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# ENHANCED RAG ENDPOINTS (Risk Prediction)
# ============================================

@router.post("/predict/explain", response_model=RiskExplanationResponse)
async def explain_risk(request: CuttingFeaturesRequest):
    """
    Explain why a waste risk was predicted.
    Combines ML prediction with knowledge base context.
    """
    try:
        enhanced_service = get_enhanced_rag_service()
        features = request.model_dump()
        result = await enhanced_service.explain_risk(features)

        sources = [SourceDocument(**s) for s in result.get("sources", [])]

        return RiskExplanationResponse(
            prediction=result["prediction"],
            risk_level=result["risk_level"],
            explanation=result["explanation"],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/recommend", response_model=RecommendationsResponse)
async def get_recommendations(request: CuttingFeaturesRequest):
    """
    Get actionable cutting recommendations to reduce waste.
    """
    try:
        enhanced_service = get_enhanced_rag_service()
        features = request.model_dump()
        result = await enhanced_service.get_recommendations(features)
        return RecommendationsResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/alert", response_model=AlertResponse)
async def get_alert(request: CuttingFeaturesRequest):
    """
    Get a one-line alert for dashboard display.
    """
    try:
        enhanced_service = get_enhanced_rag_service()
        features = request.model_dump()
        result = await enhanced_service.get_alert(features)
        return AlertResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Check if RAG system is healthy"""
    enhanced_service = get_enhanced_rag_service()
    return {
        "status": "healthy",
        "basic_rag": "available",
        "enhanced_rag": enhanced_service.client is not None,
        "vector_store": get_vector_store() is not None
    }


@router.get("/predict/health")
async def predict_health():
    """Health check specifically for enhanced RAG"""
    enhanced_service = get_enhanced_rag_service()
    return {
        "status": "healthy",
        "service": "enhanced-rag",
        "gemini_available": enhanced_service.client is not None
    }