import pandas as pd
from joblib import load

model_rest= load("app/artifacts/model_rest.joblib")
model_young= load("app/artifacts/model_young.joblib")



scaler_rest= load("app/artifacts/scaler_rest.joblib")
scaler_young= load("app/artifacts/scaler_young.joblib")


def predict():
    pass