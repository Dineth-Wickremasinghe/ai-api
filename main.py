# main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Import ALL routers
from routers import rag, retrain


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up Cutting RAG System...")

    # Try to load model
    try:
        from model import load_model
        load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Model not loaded: {e}")

    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan, title="Cutting RAG System")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include ALL routers
app.include_router(rag.router)
app.include_router(retrain.router)


@app.get("/")
async def root():
    return {
        "message": "Complete Cutting RAG System is running",
        "endpoints": {
            "enhanced_rag": "/enhanced-rag/explain, /enhanced-rag/recommend, /enhanced-rag/alert",
            "rag": "/rag/ingest, /rag/query",
            "retrain": "/retrain"
        }
    }


@app.get("/health")
async def health():
    from model import get_model
    return {"status": "ok", "model_loaded": get_model() is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)