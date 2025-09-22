# app.py

import streamlit as st
from prediction import load_model, predict_fraud, FEATURE_NAMES, DEFAULT_THRESHOLD
import pandas as pd
import base64
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Advanced Styling ---
st.markdown("""
<style>
    /* Import a nice font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* General body styling */
    body {
        font-family: 'Poppins', sans-serif;
    }

    /* Main content area styling for readability */
    [data-testid="stAppViewContainer"] > .main {
        background-color: rgba(0, 0, 0, 0.6); /* Semi-transparent black overlay */
        padding: 2rem;
        border-radius: 15px;
    }

    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 600;
        text-align: center;
        color: #FFFFFF;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }

    /* Header styling */
    h2 {
        color: #00BFFF; /* Deep Sky Blue */
        border-bottom: 2px solid #00BFFF;
        padding-bottom: 10px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 20, 0.8);
        border-right: 2px solid #00BFFF;
    }

    /* Input widget styling */
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 5px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #00BFFF;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #009ACD;
    }
    
    /* Expander styling */
    .stExpander {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid #00BFFF;
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# --- Background Image Function ---
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    st.markdown(
        f"<style>.stApp {{ background-image: url('data:image/png;base64,{img_data}'); background-size: cover; }}</style>",
        unsafe_allow_html=True,
    )

set_background("assets/background.png")

# --- Model Loading (Cached for Performance) ---
@st.cache_resource
def cached_load_model():
    return load_model()

model = cached_load_model()

if model is None:
    st.error("Model 'xgb_model.pkl' not found. Please ensure it's in the root directory.")
    st.stop()

# --- HEADER SECTION ---
st.markdown("<h1 class='main-title'>🚀 Real-Time Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.write("---")

# --- INPUT & PREDICTION SECTION ---
with st.container(border=True):
    st.header("Enter Transaction Details")
    
    # Using 4 columns for a more compact layout
    cols = st.columns(4)
    input_data = {}
    
    ordered_features = [
        'tx_amount', 'high_value_flag', 'hour', 'day_of_week', 'customer_id_enc', 'terminal_id_enc',
        'customer_txn_count', 'customer_avg_amount', 'customer_fraud_rate', 'terminal_txn_count', 
        'terminal_avg_amount', 'terminal_fraud_rate', 'customer_txn_rolling_7d', 'customer_fraud_rolling_7d',
        'terminal_txn_rolling_7d', 'terminal_fraud_rolling_7d', 'customer_fraud_14d', 'terminal_fraud_28d'
    ]
    
    for idx, feat in enumerate(ordered_features):
        with cols[idx % 4]:
            input_data[feat] = st.number_input(
                feat.replace('_', ' ').title(), 
                min_value=0.0 if 'rate' in feat or 'amount' in feat else 0,
                format="%.4f" if 'rate' in feat or 'amount' in feat else "%d",
                key=feat
            )

    # Prediction Button
    if st.button("Analyze Transaction", key="predict_button"):
        with st.spinner('Running fraud analysis...'):
            time.sleep(1) # Simulate processing time
            label, proba = predict_fraud(model, input_data, st.session_state.threshold)

        st.subheader("Prediction Result")
        result_cols = st.columns([3, 1])
        with result_cols[0]:
            if label == 1:
                st.error("🚨 **High Risk: Fraud Detected!**")
            else:
                st.success("✅ **Low Risk: Transaction is Legitimate.**")
        
        with result_cols[1]:
            st.metric(label="Fraud Probability", value=f"{proba:.2%}", delta=None)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
st.session_state.threshold = st.sidebar.slider(
    "Prediction Threshold", 0.50, 1.00, DEFAULT_THRESHOLD, 0.01,
    help="Adjust the confidence level for fraud detection. Higher values reduce false positives but might miss some fraud cases."
)
st.sidebar.info(f"Using default tuned threshold: {DEFAULT_THRESHOLD}")
st.sidebar.write("---")
st.sidebar.write("This dashboard uses an XGBoost model to predict fraudulent transactions in real-time.")

# --- MODEL INSIGHTS SECTION ---
st.header("📊 Model Performance Insights")
st.write("The following plots show the model's performance on the test dataset.")

plots = [
    ("roc_curve.png", "ROC Curve", "Shows the model's ability to distinguish between classes."),
    ("pr_curve_f1_threshold.png", "Precision-Recall & F1 Score vs. Threshold", "Helps in choosing a threshold that balances precision and recall."),
    ("confusion_matrix.png", "Confusion Matrix", "Summarizes prediction accuracy for each class."),
    ("feature_importance.png", "Feature Importance", "Ranks features by their impact on predictions."),
    ("shap_summary_bar.png", "SHAP Summary (Bar)", "Shows the average impact of each feature on model output."),
    ("shap_summary_dot.png", "SHAP Summary (Dot)", "Illustrates how feature values drive individual predictions."),
]

# Use expanders to keep the layout clean and organized
for filename, caption, help_text in plots:
    with st.expander(f"View {caption}"):
        st.image(f"assets/{filename}", use_container_width=True, caption=caption)
        st.info(help_text)