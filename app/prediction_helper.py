# codebasics ML course: codebasics.io, all rights reserved

import pandas as pd
import joblib
import os

# --- Centralized artifact path ---
BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, "..", "artifacts")

# --- Load models and scalers ---
model_young = joblib.load(os.path.join(ARTIFACT_DIR, "model_young.joblib"))
model_rest = joblib.load(os.path.join(ARTIFACT_DIR, "model_rest.joblib"))
scaler_young = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_young.joblib"))
scaler_rest = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_rest.joblib"))


def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)
    max_score = 14
    normalized_risk_score = total_risk_score / max_score
    return normalized_risk_score


def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)
    return df


def handle_scaling(age, df):
    # Choose correct scaler
    scaler_object = scaler_young if age <= 25 else scaler_rest

    try:
        # Detect and unpack scaler format
        if isinstance(scaler_object, dict):
            scaler = scaler_object.get("scaler")
            cols_to_scale = scaler_object.get("cols_to_scale", [])
        elif isinstance(scaler_object, list) and len(scaler_object) >= 2:
            scaler = scaler_object[0]
            cols_to_scale = scaler_object[1]
        else:
            scaler = None
            cols_to_scale = []

        # Only apply scaling if a valid scaler is found
        if scaler is not None and hasattr(scaler, "transform"):
            df['income_level'] = None  # temporary column for shape compatibility
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
            df.drop('income_level', axis='columns', inplace=True)
        else:
            print(f"⚠️ Skipping scaling: invalid scaler type ({type(scaler)})")

    except Exception as e:
        print(f"⚠️ Skipping scaling due to error: {e}")

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)
    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)
    return int(prediction[0])
