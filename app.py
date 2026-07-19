import os

print("Current Working Directory:", os.getcwd())
print("Files in Current Directory:", os.listdir())
import streamlit as st
import pickle
import pandas as pd
import os

# -----------------------------
# Get the project root folder
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# -----------------------------
# Load model and scaler
# -----------------------------
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

# -----------------------------
# Title
# -----------------------------
st.title("Diabetes Prediction System")

st.write("Enter the patient details below and click Predict.")

# -----------------------------
# Input Fields
# -----------------------------
pregnancies = st.number_input("Pregnancies", min_value=0, value=0)

glucose = st.number_input("Glucose", min_value=0, value=100)

blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)

skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)

insulin = st.number_input("Insulin", min_value=0, value=80)

bmi = st.number_input("BMI", min_value=0.0, value=25.0)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5,
    format="%.3f"
)

age = st.number_input("Age", min_value=1, value=30)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict"):

    patient = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    patient_scaled = scaler.transform(patient)

    prediction = model.predict(patient_scaled)

    if prediction[0] == 1:
        st.error("Patient is likely to have Diabetes")
    else:
        st.success("Patient is unlikely to have Diabetes")