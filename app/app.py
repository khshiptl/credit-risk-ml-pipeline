import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("Credit Risk Prediction")

model_path = Path("models/model.joblib")

if not model_path.exists():
    st.write("No trained model found. Please train a model first.")
else:
    model = joblib.load(model_path)
    st.write("Model loaded successfully. Enter values to predict default risk.")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=10000, max_value=200000, value=50000)
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=50000, value=10000)

    if st.button("Predict"):
        df = pd.DataFrame(
            [{"age": age, "income": income, "loan_amount": loan_amount}]
        )
        prob = model.predict_proba(df)[0, 1]
        st.write(f"Predicted default probability: {prob:.2%}")

