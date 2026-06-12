import joblib
import pandas as pd
import numpy as np

# Load model, scaler, and expected columns
model_path = "models/model.pkl"
scaler_path = "models/model_scaler.pkl"
cols_path = "models/model_cols.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
expected_cols = joblib.load(cols_path)

# Let's define a function to test an input dict
def get_prediction(input_data):
    df_input = pd.DataFrame([input_data])
    df_input_prep = pd.get_dummies(df_input, drop_first=True)
    
    for col in expected_cols:
        if col not in df_input_prep.columns:
            df_input_prep[col] = 0
    df_input_prep = df_input_prep[expected_cols]
    
    numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    df_scaled = df_input_prep.copy()
    df_scaled[numeric_columns] = scaler.transform(df_scaled[numeric_columns])
    
    y_prob = model.predict_proba(df_scaled)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)
    return y_pred[0], y_prob[0]

# Let's test two scenarios
# Scenario 1: Typical subscriber (high duration, moderate/high balance, low campaign contacts)
sub_input = {
    'age': 35.0,
    'job': 'management',
    'marital': 'single',
    'education': 'tertiary',
    'default': 'no',
    'balance': 2500.0,
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'oct',
    'duration': 800.0,  # long call duration is highly correlated with subscription
    'campaign': 1.0,
    'pdays': 90.0,
    'previous': 1.0,
    'poutcome': 'success'
}

# Scenario 2: Typical non-subscriber (short duration, low balance, housing loan, high campaign contacts)
non_sub_input = {
    'age': 45.0,
    'job': 'blue-collar',
    'marital': 'married',
    'education': 'secondary',
    'default': 'no',
    'balance': 50.0,
    'housing': 'yes',
    'loan': 'yes',
    'contact': 'unknown',
    'day': 20,
    'month': 'may',
    'duration': 50.0,  # very short duration
    'campaign': 6.0,
    'pdays': -1.0,
    'previous': 0.0,
    'poutcome': 'unknown'
}

pred_sub, prob_sub = get_prediction(sub_input)
pred_non, prob_non = get_prediction(non_sub_input)

print(f"Subscribed (yes) test: prediction={pred_sub}, probability={prob_sub:.4f}")
print(f"Not Subscribed (no) test: prediction={pred_non}, probability={prob_non:.4f}")

# Let's print the clean form values for the user
print("\n--- FORM VALUES FOR 'YES' (BERLANGGANAN) ---")
for k, v in sub_input.items():
    print(f"{k}: {v}")

print("\n--- FORM VALUES FOR 'NO' (TIDAK BERLANGGANAN) ---")
for k, v in non_sub_input.items():
    print(f"{k}: {v}")
