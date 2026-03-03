from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
from database import get_db
from db_models import Prediction
from model import retrain_model
import json
import os

router = APIRouter(prefix="/retrain", tags=["Retrain"])

FEATURE_COLUMNS = [
    "Pattern_Complexity",
    "Operator_Experience_Years",
    "Fabric_Pattern_encoded",
    "Cutting_Method_Manual",
    "Fabric_Type_encoded",
    "Marker_Loss_pct",
]

TARGET_COLUMN = "Fabric_Wastage_pct"
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Load encoding maps once at startup
with open(os.path.join(BASE_DIR, "target_encoding_mappings.json")) as f:
    encodings = json.load(f)

fabric_type_map    = encodings["Fabric_Type"]
fabric_pattern_map = encodings["Fabric_Pattern"]


@router.post("")
def trigger_retrain(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    result = _run_retrain(db)
    return result


def _run_retrain(db: Session):
    records = db.query(Prediction).filter(
        Prediction.actual_result.isnot(None)
    ).all()

    if len(records) < 5: #change back to 5 for testing
        print(f"[Retrain] Not enough records ({len(records)}). Skipping.")
        return




    GLOBAL_MEAN = np.mean([r.actual_result for r in records])


    X, y = [], []
    for r in records:
        try:
            features = [
                float(r.pattern_complexity),
                float(r.operator_experience),
                fabric_pattern_map.get(r.fabric_pattern, GLOBAL_MEAN),
                float(r.cutting_method),
                fabric_type_map.get(r.fabric_type, GLOBAL_MEAN),
                float(r.marker_loss_pct),
            ]

            X.append(features)
            y.append(r.actual_result)

        except Exception as e:
            print("Skipping row:", e)
    if not X:
        return {"status": "error", "message": "No valid rows after processing."}

    original_df = pd.read_csv(os.path.join(BASE_DIR, "processed_fabric_data.csv"))

    new_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    new_df[TARGET_COLUMN] = y

    # Combine
    combined_df = pd.concat([original_df, new_df], ignore_index=True)

    X_combined = combined_df[FEATURE_COLUMNS]
    y_combined = combined_df[TARGET_COLUMN].values

    replaced, new_score, old_score = retrain_model(X_combined, y_combined)  # fix: was passing X/y instead of combined

    print("Training rows:", len(X))
    if replaced:
        return {
            "status": "improved",
            "message": f"Model updated! R² improved from {old_score:.4f} to {new_score:.4f}",
            "old_score": old_score,
            "new_score": new_score
        }
    else:
        return {
            "status": "no_improvement",
            "message": f"No improvement. Current: {old_score:.4f} | New: {new_score:.4f}",
            "old_score": old_score,
            "new_score": new_score
        }