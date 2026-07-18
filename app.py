import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("logistic_model.pkl")

# Load scaler if available
try:
    scaler = joblib.load("scaler.pkl")
    scaler_used = True
except:
    scaler_used = False

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction using Logistic Regression")

st.write("Enter the patient's details below:")

Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose")
BloodPressure = st.number_input("Blood Pressure")
SkinThickness = st.number_input("Skin Thickness")
Insulin = st.number_input("Insulin")
BMI = st.number_input("BMI")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
Age = st.number_input("Age", min_value=1)

if st.button("Predict"):

    data = np.array([[Pregnancies,
                      Glucose,
                      BloodPressure,
                      SkinThickness,
                      Insulin,
                      BMI,
                      DiabetesPedigreeFunction,
                      Age]])

    if scaler_used:
        data = scaler.transform(data)

    prediction = model.predict(data)

    probability = model.predict_proba(data)

    if prediction[0] == 1:
        st.error("Prediction: Diabetic")
    else:
        st.success("Prediction: Not Diabetic")

    st.write("Prediction Probability")
    st.write(probability)
