
import streamlit as st
import pickle
import numpy as np

st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict the risk of heart disease.")

# Load Model
model = pickle.load(open("heart_model.pkl", "rb"))

# User Inputs
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol Level", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0,1])
restecg = st.slider("Resting ECG Results (0-2)", 0, 2, 1)
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0,1])
oldpeak = st.number_input("ST Depression Induced", 0.0, 10.0, 1.0)
slope = st.slider("Slope (0-2)", 0, 2, 1)
ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.selectbox("Thal (1-3)", [1,2,3])

# Prepare Input
features = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("✅ High risk of Heart Disease. Consult a doctor.")
    else:
        st.info("🟢 No major risk detected.")
