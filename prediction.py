# prediction.py

import pickle
import numpy as np

# Standardized feature names (all lowercase, no underscores) for consistency.
FEATURE_NAMES = [
    'tx_amount', 'hour', 'day_of_week', 'customer_txn_count', 'customer_avg_amount',
    'customer_fraud_rate', 'terminal_txn_count', 'terminal_avg_amount', 'terminal_fraud_rate',
    'high_value_flag', 'customer_txn_rolling_7d', 'customer_fraud_rolling_7d',
    'terminal_txn_rolling_7d', 'terminal_fraud_rolling_7d', 'terminal_fraud_28d',
    'customer_fraud_14d', 'customer_id_enc', 'terminal_id_enc'
]

# Set the default threshold found during model tuning
DEFAULT_THRESHOLD = 0.97

def load_model(model_path='xgb_model.pkl'):
    """
    Loads the trained XGBoost model from a pickle file.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

def predict_fraud(model, input_data, threshold=DEFAULT_THRESHOLD):
    """
    Makes a fraud prediction based on the input data and a given threshold.
    """
    # Create the feature vector in the correct order
    X = np.array([[input_data[feat] for feat in FEATURE_NAMES]])

    # Get the probability of fraud (class 1)
    proba = model.predict_proba(X)[0, 1]

    # Determine the label based on the threshold
    label = int(proba >= threshold)

    return label, proba