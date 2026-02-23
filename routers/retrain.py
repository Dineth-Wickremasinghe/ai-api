from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session
import numpy as np

from database import get_db
from db_models import Prediction
from model import retrain_model

router = APIRouter(prefix="/retrain", tags=["Retrain"])


@router.post("")
def trigger_retrain(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    background_tasks.add_task(_run_retrain, db)
    return {"message": "Retraining started in the background."}


def _run_retrain(db: Session):
    records = db.query(Prediction).filter(
        Prediction.actual_result.isnot(None)
    ).all()

    if len(records) < 50: #change back to 5 for testing
        print(f"[Retrain] Not enough records ({len(records)}). Skipping.")
        return

    # Parse comma-separated input_features back into feature vectors
    X, y = [], []
    for r in records:
        try:
            cleaned = r.input_features.strip("[]")
            features = [float(v) for v in  cleaned.split(",")]
            if len(features) == 6:
                X.append(features)
                y.append(r.actual_result)
        except Exception:
            continue

    if len(X) < 50: #change back to 5 for testing
        print("[Retrain] Not enough parseable records. Skipping.")
        return

    replaced, score = retrain_model(np.array(X), np.array(y))

    if replaced:
        print(f"[Retrain] Model replaced. New R²: {score:.4f}")
    else:
        print(f"[Retrain] New model did not improve. Kept current. R²: {score:.4f}")