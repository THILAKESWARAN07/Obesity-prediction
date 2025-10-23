import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load("xgboost.pkl")

# Streamlit app
st.set_page_config(page_title="Obesity Category Prediction", page_icon="🏃", layout="centered")
st.title("🏃 Obesity Category Prediction App")

st.markdown("""
This app predicts a person's **Obesity Category** (Normal weight, Overweight, Obese, etc.)  
based on physical characteristics and activity level using an **XGBoost model**.
""")

# --- Input fields ---
st.header("Enter the following details:")

age = st.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (in cm)", min_value=100.0, max_value=250.0, value=170.0, format="%.2f")
weight = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.2f")
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, format="%.2f")
activity = st.slider("Physical Activity Level (1 = Low, 5 = High)", 1, 5, 3)

# --- Encoding categorical features ---
gender_encoded = 1 if gender == "Male" else 0  # Male=1, Female=0

# --- Combine features into array ---
features = np.array([[age, gender_encoded, height, weight, bmi, activity]])

# --- Prediction ---
if st.button("Predict Obesity Category"):
    prediction = model.predict(features)
    st.success(f"🏆 Predicted Obesity Category: **{prediction[0]}**")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and XGBoost")
