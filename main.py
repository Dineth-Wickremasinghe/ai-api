from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from model import load_model, get_model
from typing import List
from routers import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(rag.router)


class PredictionRequest(BaseModel):
    features: List[float]   # e.g. [5.1, 3.5, 1.4, 0.2]


class PredictionResponse(BaseModel):
    prediction: float | int | str
    confidence: float | None = None  # optional


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
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

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model() is not None}