import pandas as pd
import numpy as np


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

    df = pd.get_dummies(df, drop_first=True, dtype=int)

    
    return df
