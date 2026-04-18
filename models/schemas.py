# models/schemas.py

from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    collection: Optional[str] = None
    role: Optional[str] = None

class SourceDocument(BaseModel):
    source: Optional[str]
    page: Optional[int]

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_stored: int

# ========== Enhanced RAG Schemas ==========

class CuttingFeaturesRequest(BaseModel):
    pattern_complexity: float
    operator_experience: float
    fabric_pattern: str
    fabric_type: str
    cutting_method: int
    marker_loss_pct: float
    fabric_pattern_encoded: Optional[float] = None
    fabric_type_encoded: Optional[float] = None

class RiskExplanationResponse(BaseModel):
    prediction: float
    risk_level: str
    explanation: str
    sources: List[SourceDocument]

class RecommendationsResponse(BaseModel):
    prediction: float
    risk_level: str
    immediate_action: str
    process_change: str
    expected_improvement: str
    confidence: str
    based_on: str

class AlertResponse(BaseModel):
    alert: str