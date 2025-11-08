import streamlit as st
import pandas as pd
from prediction_helper import predict

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.title("🏥 Health Insurance Input Form")

st.write("Enter the details below. The app will construct a DataFrame from your inputs.")

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    region = st.selectbox("Region", ['Northwest', 'Southeast', 'Northeast', 'Southwest'])
    marital_status = st.selectbox("Marital Status", ['Unmarried', 'Married'])
    bmi_category = st.selectbox("BMI Category", ['Normal', 'Obesity', 'Overweight', 'Underweight'])
    smoking_status = st.selectbox("Smoking Status", ['No Smoking', 'Regular', 'Occasional'])

with col2:
    employment_status = st.selectbox("Employment Status", ['Salaried', 'Self-Employed', 'Freelancer'])
    income_level = st.selectbox("Income Level", [1, 2, 3, 4])
    medical_history = st.selectbox("Medical History", [
        'No Disease', 'Diabetes', 'High blood pressure',
        'Diabetes & High blood pressure', 'Thyroid', 'Heart disease',
        'High blood pressure & Heart disease', 'Diabetes & Thyroid',
        'Diabetes & Heart disease'
    ])
    insurance_plan = st.selectbox("Insurance Plan", [1, 2, 3])

# --- Create DataFrame from inputs ---
input_dict = {
    'gender': [gender],
    'region': [region],
    'marital_status': [marital_status],
    'bmi_category': [bmi_category],
    'smoking_status': [smoking_status],
    'employment_status': [employment_status],
    'income_level': [income_level],
    'medical_history': [medical_history],
    'insurance_plan': [insurance_plan]
}

input_df = pd.DataFrame(input_dict)

# --- Show user input data ---
st.subheader("🧾 Constructed DataFrame")
st.dataframe(input_df)

# --- Footer ---
st.markdown("---")
st.caption("💬 This app only collects inputs and displays them as a DataFrame. You can later connect it to a model for predictions.")
