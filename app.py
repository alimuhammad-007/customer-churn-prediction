
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "customer_churn_xgboost.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict customer churn risk using a trained XGBoost model."
)

try:
    artifact = load_model()
    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_names = artifact["features"]
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.info(f"Current classification threshold: {threshold:.2f}")

# Customer inputs
st.subheader("Customer Information")

# These defaults are illustrative; replace with valid
# dataset categories if your dataset differs.
defaults = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x else "No"
        )
        partner = st.selectbox(
            "Partner", ["Yes", "No"]
        )
        dependents = st.selectbox(
            "Dependents", ["Yes", "No"]
        )
        tenure = st.number_input(
            "Tenure (months)", 0, 72, 12
        )
        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0,
            value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0,
            value=840.0, step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([customer])

    # Ensure exact feature order expected by model
    input_df = input_df[feature_names]

    probability = model.predict_proba(input_df)[0, 1]
    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

    with c2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.0%}"
        )

    st.progress(float(probability))

    if prediction == 1:
        st.error("High Churn Risk — retention review recommended.")
    else:
        st.success("Lower Churn Risk — continue monitoring.")

    st.caption(
        "This is a model estimate, not a guarantee of future behavior."
    )
