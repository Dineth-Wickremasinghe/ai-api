# AI API — FastAPI ML Prediction + RAG Service

A FastAPI application that serves two purposes:
1. **ML Predictions** — runs a scikit-learn model and returns predictions to a Spring Boot backend
2. **RAG (Retrieval-Augmented Generation)** — ingests PDF documents, stores embeddings in ChromaDB, and answers questions using Google Gemini

---

## Architecture

```
Client / Frontend
       │
       ▼
 Spring Boot App
  ┌────┴─────┐
  │          │
  ▼          ▼
/predict   /rag/query
           /rag/ingest
       │
       ▼
  FastAPI (this service)
  ┌────┴──────────────┐
  │                   │
  ▼                   ▼
scikit-learn       ChromaDB (vector store)
ML Model           + Google Gemini (LLM + Embeddings)
```

Spring Boot acts as the gateway — it forwards requests to this FastAPI service which handles all ML inference and AI/RAG logic.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| ML Inference | scikit-learn |
| LLM | Google Gemini (`gemini-2.0-flash`) |
| Embeddings | Google Gemini (`models/gemini-embedding-001`) |
| Vector Database | ChromaDB (local, file-based) |
| RAG Framework | LangChain (LCEL) |
| PDF Parsing | PyMuPDF (fitz) |
| Runtime | Python 3.11 |

---

## Project Structure

```
ai-api/
├── main.py                  # App entrypoint, registers all routers
├── config.py                # Settings loaded from .env
├── model.py                 # ML model loader
├── routers/
│   ├── predict.py           # POST /predict
│   └── rag.py               # POST /rag/ingest, POST /rag/query, DELETE /rag/collection
├── services/
│   ├── pdf_service.py       # PDF parsing and text chunking
│   ├── vector_store.py      # ChromaDB + Gemini embeddings setup
│   └── rag_service.py       # RAG chain logic (LangChain LCEL)
├── models/
│   └── schemas.py           # Pydantic request/response schemas
├── chroma_db/               # Auto-created — persistent vector store (gitignore this)
├── .env                     # Environment variables (never commit this)
├── .env.example             # Example env file (safe to commit)
└── requirements.txt
```

---

## Prerequisites

- Python **3.11** (required — 3.12+ has compatibility issues with ChromaDB/Pydantic V1)
- A **Google Gemini API key** — get one at [https://aistudio.google.com](https://aistudio.google.com)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai-api.git
cd ai-api
```

### 2. Create and activate a virtual environment

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
COLLECTION_NAME=documents
EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_MODEL=gemini-2.0-flash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 5. Run the app

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
Interactive docs (Swagger UI) at `http://localhost:8000/docs`

---

## API Endpoints

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### ML Prediction

```
POST /predict
```

Request:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:
```json
{
  "prediction": 0,
  "confidence": 0.97
}
```

---

### Ingest a PDF

```
POST /rag/ingest
Content-Type: multipart/form-data
```

Upload a PDF file. The service will parse, chunk, embed, and store it in ChromaDB.

```bash
curl -X POST http://localhost:8000/rag/ingest \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "message": "PDF ingested successfully.",
  "filename": "your_document.pdf",
  "chunks_stored": 35
}
```

---

### Query Documents

```
POST /rag/query
Content-Type: application/json
```

Ask a question — Gemini will answer using only the content from your ingested PDFs.

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

Response:
```json
{
  "answer": "Based on the document, it discusses...",
  "sources": [
    { "source": "your_document.pdf", "page": 1 },
    { "source": "your_document.pdf", "page": 4 }
  ]
}
```

---

### Clear Vector Store

```
DELETE /rag/collection
```

Wipes all stored document embeddings from ChromaDB. Use with caution.

---

## Bulk Ingesting PDFs

To load a folder of PDFs at once instead of uploading one by one:

```bash
python scripts/bulk_ingest.py ./pdfs
```

Place all your PDF files in a `./pdfs` folder before running.

---

## Spring Boot Integration

This service is designed to be called from a Spring Boot backend. See the Spring Boot side for `FastApiService.java` which uses `WebClient` to forward requests.

**Ingest a PDF from Spring Boot:**
```java
MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("file", new FileSystemResource(pdfFile));
restTemplate.postForObject("http://localhost:8000/rag/ingest", body, IngestResponse.class);
```

**Query from Spring Boot:**
```java
QueryRequest req = new QueryRequest("What does the document say about pricing?");
QueryResponse resp = restTemplate.postForObject("http://localhost:8000/rag/query", req, QueryResponse.class);
```

---

## How RAG Works

1. **Ingest** — PDF is parsed page by page using PyMuPDF, split into overlapping chunks using `RecursiveCharacterTextSplitter`, embedded using `models/gemini-embedding-001`, and stored in a local ChromaDB collection
2. **Query** — The question is embedded and compared against stored chunks using MMR (Maximal Marginal Relevance) retrieval to find the most relevant, diverse context
3. **Generate** — The retrieved chunks are passed as context to Gemini, which generates a grounded answer and returns it along with source citations

---

## Environment Variables Reference

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key | required |
| `COLLECTION_NAME` | ChromaDB collection name | `documents` |
| `EMBEDDING_MODEL` | Gemini embedding model | `models/gemini-embedding-001` |
| `GEMINI_MODEL` | Gemini chat model | `gemini-2.5-flash` |
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

---

## .gitignore Recommendations

Make sure these are in your `.gitignore`:

```
.env
.venv/
chroma_db/
__pycache__/
*.pyc
*.pkl       # trained ML model files
```

---

## Requirements

```
fastapi
uvicorn[standard]
langchain
langchain-core
langchain-google-genai
langchain-community
langchain-chroma
langchain-text-splitters
chromadb
google-generativeai
google-genai
pypdf
pymupdf
python-multipart
pydantic-settings
scikit-learn
```

---

## Troubleshooting

**`ModuleNotFoundError` on startup** — make sure your venv is activated and you ran `pip install -r requirements.txt`

**Embedding 404 error** — verify your `GEMINI_API_KEY` is valid and `EMBEDDING_MODEL` is set to `models/gemini-embedding-001`

**ChromaDB Pydantic error** — you are likely on Python 3.12+. Downgrade to Python 3.11

**`pip` not recognized** — your venv is not activated. Run `.venv\Scripts\activate` first (Windows)
