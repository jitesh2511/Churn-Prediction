import pandas as pd
import joblib
import sys

from src.config import MODEL_DIR, THRESHOLD
from src.preprocess import preprocess_inf

def predict(data):

    """
    The predict function serves as a pipeline that takes in input data (as a Pandas DataFrame),
    preprocesses it to match the feature set and format expected by the trained model, 
    loads the trained model, and produces predictions. 

    Specifically, it performs the following steps:
    1. Loads a trained model from disk using joblib.
    2. Preprocesses the input data using the preprocess_inf function to ensure
       correct feature types, encodings, and scaling, matching what the model expects.
    3. Removes the target variable 'Churn' if present to avoid data leakage.
    4. Applies the model to generate predicted probabilities and classes for churn.
    5. Returns both the predictions (as binary 0/1) and the corresponding probabilities.

    This function enables seamless batch or single-row inference for new/unseen data.
    """


    # Load Model
    model = joblib.load(MODEL_DIR/"model.pkl")
    required_cols = joblib.load(MODEL_DIR/"model_columns.pkl")

    # Adding all columns before preprocessing
    for col in required_cols:
        if col not in data.columns:
            data[col] = 0

    # Preprocess
    df = preprocess_inf(data)

    # Dropping target variable if present
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)

    # Predict
    y_prob = model.predict_proba(df)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    return y_pred, y_prob
