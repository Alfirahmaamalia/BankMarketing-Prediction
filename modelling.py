import sys
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib

# Fix stdout encoding for mlflow emojis on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Configuration
DAGSHUB_USERNAME = "Alfirahmaamalia"
DAGSHUB_TOKEN = "0c17740769ba0ad2bbd3571d3cbc214816579a02"
# MLflow Authentication Injection
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

MLFLOW_TRACKING_URI = f"https://dagshub.com/Alfirahmaamalia/BankMarketing-Prediction.mlflow"
EXPERIMENT_NAME = "BankMarketing-Prediction"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.getenv("DATA_PATH", os.path.join(BASE_DIR, "bank_marketing_final.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "model.pkl"))

def setup_mlflow():
    """Setup tracking URI to DagsHub"""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"MLflow connected to: {MLFLOW_TRACKING_URI}")
    except Exception as e:
        print(f"Warning: Failed to connect to MLflow: {e}")

def load_data(data_path: str = DATA_PATH):
    """Load dataset"""
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded: {data_path}")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None

def preprocess_data(df):
    """Preprocessing and feature engineering as per Colab"""
    try:
        df_processed = df.copy()
        
        # Drop extra columns that might have been saved in the final csv
        cols_to_drop = ['subscription_probability', 'prediction_label', 'Subscription_Status']
        df_processed = df_processed.drop(columns=[c for c in cols_to_drop if c in df_processed.columns])
        
        # OHE
        df_prep = pd.get_dummies(df_processed, drop_first=True)
        
        target_column = 'y_yes'
        if target_column not in df_prep.columns:
            print(f"Warning: Target column '{target_column}' not found. Make sure target is 'y' with yes/no.")
            return None, None, None, None
            
        X = df_prep.drop(target_column, axis=1)
        y = df_prep[target_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y, df_processed.columns.tolist(), X.columns.tolist()
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def train_model(X, y, test_size=0.2, random_state=42):
    """Train Random Forest model with MLflow Tracking to DagsHub"""
    try:
        import mlflow
        import mlflow.sklearn
        setup_mlflow()
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        # Split data (Colab used stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scaling only numeric columns
        scaler = StandardScaler()
        numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
        
        with mlflow.start_run(run_name="RandomForest_Colab_Config"):
            # Log params
            mlflow.log_params({
                "model_type": "RandomForest",
                "n_estimators": 100,
                "class_weight": "balanced",
                "test_size": test_size,
                "random_state": random_state,
                "input_features": len(X.columns),
                "threshold": 0.3
            })

            # Train model
            print("Training Random Forest model (Colab config)...")
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced'
            )
            model.fit(X_train_scaled, y_train)
            
            # Predict probabilities with custom threshold 0.3
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_prob >= 0.3).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }
            mlflow.log_metrics(metrics)
            
            # Run Description
            mlflow.set_tag("mlflow.note.content", "Random Forest model configured for Business Case. class_weight='balanced' with Custom Threshold 0.3 for High Recall.")
            
            # Save model to MLflow
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_test_scaled, y_pred)
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name="bank_marketing_model"
            )
            
            # Save model locally for Flask Web App
            os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            
            # Save scaler
            scaler_path = MODEL_PATH.replace(".pkl", "_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            
            # Save expected columns
            cols_path = MODEL_PATH.replace(".pkl", "_cols.pkl")
            joblib.dump(X.columns.tolist(), cols_path)
            
            print(f"\n{'='*50}")
            print(f"Model Training Complete!")
            print(f"Log pushed to DagsHub Experiments!")
            print(f"{'='*50}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC AUC:   {roc_auc:.4f}")
            print(f"{'='*50}")
            
            return model, scaler, metrics
            
    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def predict(input_data):
    """
    Make predictions using the trained model
    """
    try:
        from model_util import get_model
        import pandas as pd
        import joblib
        
        model = get_model(use_local=True)
        if model is None:
            return "Error: Model not found"
        
        # Convert input to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # OHE
        df_input_prep = pd.get_dummies(df_input, drop_first=True)
        
        # Load scaler and expected columns
        scaler_path = MODEL_PATH.replace(".pkl", "_scaler.pkl")
        cols_path = MODEL_PATH.replace(".pkl", "_cols.pkl")
        scaler = joblib.load(scaler_path)
        expected_cols = joblib.load(cols_path)
        
        # Align columns
        for col in expected_cols:
            if col not in df_input_prep.columns:
                df_input_prep[col] = 0
        df_input_prep = df_input_prep[expected_cols]
        
        # Scale numeric columns
        numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        df_scaled = df_input_prep.copy()
        df_scaled[numeric_columns] = scaler.transform(df_scaled[numeric_columns])
        
        # Predict using threshold 0.3
        y_prob = model.predict_proba(df_scaled)[:, 1]
        y_pred = (y_prob >= 0.3).astype(int)
        
        subscription_prob = y_prob[0] * 100
        
        return {
            "prediction": "yes" if y_pred[0] == 1 else "no",
            "probability": f"{subscription_prob:.2f}%"
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Starting model training pipeline...")
    
    df = load_data()
    if df is None:
        sys.exit(1)
    
    X, y, orig_cols, final_cols = preprocess_data(df)
    if X is None:
        sys.exit(1)
    
    model, scaler, metrics = train_model(X, y)
    
    if model is not None:
        print("\nModel training successful!")
    else:
        print("\nModel training failed!")
        sys.exit(1)
