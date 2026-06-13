"""
model_util.py - Utility functions for loading and managing ML models
"""
import os
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", os.path.join(BASE_DIR, "models", "model.pkl"))

def load_model_from_local(model_path: str = None):
    """
    Load model from local file
    """
    try:
        if model_path is None:
            model_path = LOCAL_MODEL_PATH
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        
        model = joblib.load(model_path)
        print(f"Model loaded successfully from local file: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from local file: {e}")
        return None

def get_model(use_local: bool = True):
    """
    Get model from local
    """
    return load_model_from_local()

def get_feature_names():
    features = [
        'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
    ]
    return features

if __name__ == "__main__":
    # Test loading model
    print("Testing model loading...")
    
    model = get_model()
    
    if model:
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Features: {get_feature_names()}")
    else:
        print("Failed to load model")
