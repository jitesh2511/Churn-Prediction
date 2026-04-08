import pandas as pd
import joblib

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

    # Preprocess
    df = preprocess_inf(data)

    # Dropping target variable if present
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)

    # Predict
    y_prob = model.predict_proba(df)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    return y_pred, y_prob


def imp_factors(sample:pd.DataFrame):

    """
    The imp_factors function is used in app.py to provide interpretability and explainability for the model's churn prediction.

    Specifically, after making a prediction, it computes the contribution (or importance) of each input feature for a given customer sample.

    The function multiplies each input feature value by the model's learned coefficient for that feature, producing a "contribution" score
    that reflects how much that feature influenced the model's output on this sample. It returns a sorted DataFrame listing each feature,
    its contribution, and absolute contribution. In app.py, these contributions are displayed to users to explain why a customer was
    predicted as likely to churn or stay, highlighting the most influential features in the decision.
    """

    # Preprocess
    sample = preprocess_inf(sample)    

    sample = sample.iloc[0]
    model = joblib.load(MODEL_DIR/'model.pkl')
    contributions = sample * model.coef_[0]

    contrib_df = pd.DataFrame({
        'Feature': sample.index,
        'Contribution': contributions,
        'AbsContribution':abs(contributions)
    }).sort_values(by="AbsContribution", ascending=False)

    return contrib_df