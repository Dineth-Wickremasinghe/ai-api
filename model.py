import joblib


model = None

def load_model():
    global model
    model = joblib.load("fabric_waste_pipeline.pkl")
    print("Model loaded successfully")

def get_model():
    return model