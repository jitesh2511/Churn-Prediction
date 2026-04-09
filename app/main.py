from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.predict import predict, imp_factors

app = FastAPI(title='Churn Prediction API')

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str


@app.get('/')
def home():
    return {'message':'Churn Prediction API is running'}

@app.post('/predict')
def predict_churn(data: CustomerData):

    input_data = {
        'gender': data.gender,
        'SeniorCitizen': data.SeniorCitizen,
        'Partner': data.Partner,
        'Dependents': data.Dependents,
        'tenure': data.tenure,
        'MonthlyCharges': data.MonthlyCharges,
        'TotalCharges': data.TotalCharges,
        'Contract': data.Contract,
        'InternetService': data.InternetService,
        'PaymentMethod': data.PaymentMethod
    }

    df = pd.DataFrame([input_data])

    y_pred, y_prob = predict(df)

    prediction = 'Yes' if y_pred[0] == 1 else 'No'
    probability = float(round(y_prob[0] * 100, 1))

    all_features = imp_factors(df)

    # Filter
    filtered_features = []
    options = list(input_data.keys())

    for _, row in all_features.iterrows():
        for opt in options:
            if row["Feature"].startswith(opt):
                filtered_features.append(row)
                break

    filtered_df = pd.DataFrame(filtered_features)

    if filtered_df.empty:
        factors = pd.DataFrame({})

    positive = filtered_df[filtered_df["Contribution"] > 0]
    negative = filtered_df[filtered_df["Contribution"] < 0]

    if prediction == 'Yes':
        factors = positive.head(3)
               
    else:
        factors = negative.head(3)
    factors = factors.reset_index(drop=True)
    print(factors)
    print(type(factors))

    return {
        'prediction': prediction,
        'probability': probability,
        'factors': factors
    }