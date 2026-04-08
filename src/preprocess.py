import pandas as pd
from src.config import MODEL_DIR, NUMERIC_COLS
import joblib


def preprocess(df : pd.DataFrame) -> pd.DataFrame:

    """
    Preprocess Training Data

    This function performs the following preprocessing steps on the training data:
    1. Converts the 'TotalCharges' column to numeric and fills missing values with zero.
    2. Drops the 'customerID' column as it is a unique identifier and not useful for modeling.
    3. Converts all specified Yes/No columns (binary columns including the target 'Churn') to binary numerical format (1/0).
    4. Applies one-hot encoding to categorical variables (except the first category to avoid dummy variable trap).
    5. Ensures all feature columns are of float data type.
    """

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

    """
    Preprocess new/unseen data for prediction

    This function preprocesses new/unseen data before making predictions. It performs similar steps as the preprocess() function used for training,
    such as converting 'TotalCharges' to numeric, binary encoding Yes/No columns, and applying one-hot encoding.

    However, it omits any handling of the target variable 'Churn', since this column is not present in new data.
    Additionally, after encoding, it aligns the processed features to exactly match the training data columns,
    filling any missing columns with zeros. It also applies the same scaler used during training to numeric columns,
    ensuring consistency between training and inference.
    """

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

    X = pd.get_dummies(X)
    X = X.astype(float)

    # Align columns with training data
    model_columns = joblib.load(MODEL_DIR / "model_columns.pkl")
    X = X.reindex(columns=model_columns, fill_value=0)

    # Scaling the numeric columns
    scaler = joblib.load(MODEL_DIR/"scaler.pkl")

    X.loc[:, NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    return X