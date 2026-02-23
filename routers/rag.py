from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from services.pdf_service import extract_text_from_pdf, chunk_documents
from services.vector_store import get_vector_store
from services.rag_service import query_rag
from models.schemas import QueryRequest, QueryResponse, IngestResponse

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF and store its embeddings in Qdrant."""
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
    """Query the vector store and get a Gemini-powered answer."""
    try:
        result = await query_rag(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collection")
async def clear_collection():
    """Clear all documents from the vector store (use with caution)."""
    from services.vector_store import client, settings
    client.delete_collection(settings.COLLECTION_NAME)
    return {"message": "Collection cleared."}