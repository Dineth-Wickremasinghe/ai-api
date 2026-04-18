# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Import routers
from routers import rag, retrain


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up Cutting RAG System...")
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
    allow_origins=["http://localhost:8080"],  # Spring Boot port
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag.router)
app.include_router(retrain.router)


# Prediction Models
class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: float | int | str
    confidence: float | None = None


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    from model import get_model
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_data = [request.features]
        prediction = model.predict(input_data)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            confidence = float(max(proba))
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Complete Cutting RAG System is running",
        "endpoints": {
            "rag": "/rag/ingest, /rag/query",
            "retrain": "/retrain",
            "predict": "/predict"
        }
    }


@app.get("/health")
async def health():
    from model import get_model
    return {"status": "ok", "model_loaded": get_model() is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)