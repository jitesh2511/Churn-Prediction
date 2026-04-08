import pandas as pd
from src.config import MODEL_DIR
import joblib


def preprocess(df : pd.DataFrame) -> pd.DataFrame:

    # Handling the TotalCharges column

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)


    # Dropping the customerID column as it a unique identifier

    df = df.drop('customerID', axis=1)

    
    # Converting all Yes/No columns to Binary, including target column

    binary_cols = binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    
    # Converting Categorical to One-Hot Encoding

    df = pd.get_dummies(df, drop_first=True)
    df = df.astype(float)

    
    return df

def preprocess_inf(X : pd.DataFrame) -> pd.DataFrame:

    # Handling the TotalCharges column
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    X['TotalCharges'] = X['TotalCharges'].fillna(0)

    # Dropping the customerID column
    X = X.drop('customerID', axis=1, errors='ignore')

    # Converting all Yes/No columns to Binary, including target column
    binary_cols = binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].map({'Yes': 1, 'No': 0})

    # Converting Categorical to One-Hot Encoding
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)

    # Align columns with training data
    model_columns = joblib.load(MODEL_DIR / "model_columns.pkl")
    X = X.reindex(columns=model_columns, fill_value=0)

    # Scaling the numeric columns
    scaler = joblib.load(MODEL_DIR/"scaler.pkl")
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    X.loc[:, numeric_cols] = scaler.transform(X[numeric_cols])

    return X