import joblib

# Load your model once at startup, not on every request
model = None

def load_model():
    global model
    model = joblib.load("fabric_waste_pipeline.pkl")  # adjust path/format as needed
    print("Model loaded successfully")

def get_model():
    return model