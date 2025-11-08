import streamlit as st
import joblib
import pandas as pd

st.title('Heart Disease Prediction')

# Load model
model = joblib.load('heart_disease_model.pkl')

# Inputs
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex (0=Female,1=Male)', [0,1])
cp = st.number_input('Chest Pain Type (0-3)',0,3,0)
trestbps = st.number_input('Resting BP',50,250,120)
chol = st.number_input('Cholesterol',100,600,200)
fbs = st.selectbox('Fasting BS >120', [0,1])
restecg = st.number_input('Resting ECG (0-2)',0,2,0)
thalach = st.number_input('Max Heart Rate',50,250,150)
exang = st.selectbox('Exercise Angina', [0,1])
oldpeak = st.number_input('ST Depression',0.0,10.0,0.0)
slope = st.number_input('Slope (0-2)',0,2,0)
ca = st.number_input('Major Vessels (0-3)',0,3,0)
thal = st.number_input('Thalassemia (1-3)',1,3,1)

if st.button('Predict'):
    features = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error('HIGH RISK - Consult a doctor!')
    else:
        st.success('LOW RISK - Maintain a healthy lifestyle.')
