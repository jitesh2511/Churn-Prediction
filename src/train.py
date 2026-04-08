import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import DATA_PATH, MODEL_DIR, NUMERIC_COLS
from src.preprocess import preprocess

'''
This script trains a Logistic Regression model to predict customer churn.

Steps performed:

1. Loads and preprocesses the Telco Customer Churn dataset.
2. Splits data into features (X_train) and target ('Churn').
3. Applies standard scaling to numeric columns.
4. Trains a class-balanced logistic regression model.
5. Saves the trained model, feature columns, and scaler to disk for later use in inference/prediction.

Running this script produces the model artifacts required for prediction and running app.py

'''

# Loading the data
data = preprocess(pd.read_csv(DATA_PATH))


X_train = data.drop('Churn', axis=1)
y_train = data['Churn']

# Scaling
scaler = StandardScaler()

X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])

# Training the model
model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)

# Saving the model
joblib.dump(model, MODEL_DIR/'model.pkl')
joblib.dump(X_train.columns, MODEL_DIR/'model_columns.pkl')
joblib.dump(scaler, MODEL_DIR/'scaler.pkl')

print('saved the model successfully')
