import joblib
import copy
import threading
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

model = None
_model_lock = threading.Lock()

def load_model():
    global model
    model = joblib.load("fabric_waste_pipeline.pkl")
    print("Model loaded successfully")

def get_model():
    return model

def retrain_model(X, y):
    global model

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    new_model = copy.deepcopy(model)
    new_model.fit(X_train, y_train)

    new_score  = r2_score(y_test, new_model.predict(X_test))
    curr_score = r2_score(y_test, model.predict(X_test))

    if new_score > curr_score:
        joblib.dump(new_model, MODEL_PATH)
        with _model_lock:
            model = new_model
        print(f"[Retrain] Model replaced. New R²: {new_score:.4f}")
        return True, new_score
    else:
        print(f"[Retrain] No improvement. Kept current model. R²: {curr_score:.4f}")
        return False, curr_score