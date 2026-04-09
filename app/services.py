import logging
import pandas as pd

from src.predict import predict, imp_factors

def run_prediction(input_data):

    """
    Runs full inference pipeline:
    - preprocessing
    - prediction
    - feature importance extraction
    """

    logging.info("Running prediction pipeline")

    df = pd.DataFrame([input_data])

    y_pred, y_prob = predict(df)

    prediction = 'Yes' if y_pred[0] == 1 else 'No'
    probability = float(round(y_prob[0] * 100, 1))

    logging.info(f"Model output → prediction: {prediction}, prob: {probability}")

    all_features = imp_factors(df)

    filtered_features = []
    options = list(input_data.keys())

    for _, row in all_features.iterrows():
        for opt in options:
            if row["Feature"].startswith(opt):
                filtered_features.append(row)
                break

    filtered_df = pd.DataFrame(filtered_features)

    if filtered_df.empty:
        logging.warning("No important factors identified")
        factors = pd.DataFrame({})
    else:
        positive = filtered_df[filtered_df["Contribution"] > 0]
        negative = filtered_df[filtered_df["Contribution"] < 0]

        if prediction == 'Yes':
            factors = positive.head(3)
        else:
            factors = negative.head(3)

    factors = factors.reset_index(drop=True)

    return {
        'prediction': prediction,
        'probability': probability,
        'factors': factors
    }