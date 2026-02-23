from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    collection: Optional[str] = None

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