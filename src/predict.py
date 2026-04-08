import pandas as pd
import joblib
import sys

from src.config import MODEL_DIR
from src.config import DATA_PATH
from src.preprocess import preprocess_inf

def predict(data):

    # Load Model
    model = joblib.load(MODEL_DIR/"logistic_model.pkl")

    # Preprocess
    df = preprocess_inf(data)

    # Dropping target variable if present
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)

    # Predict
    y_prob = model.predict_proba(df)[:, 1]
    y_pred = (y_prob >= 0.35).astype(int)

    return y_pred, y_prob


# Uncomment to test
# run the below command to get results

# python -m src.predict data/sample.csv 

# replace data/sample.csv with your own data



# yp, ypr = predict(pd.read_csv(sys.argv[1]))

# results = pd.DataFrame({
#     "Churn Prediction": yp,
#     "Churn Probability": ypr
# })

# results["Churn Prediction"] = results["Churn Prediction"].map({
#     0: "No",
#     1: "Yes"
# })

# print(results.head(5))
