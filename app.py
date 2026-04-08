import streamlit as st
import pandas as pd
import joblib

from src.config import MODEL_DIR
from src.predict import predict

st.title("📊 Customer Churn Prediction")

st.markdown("Enter customer details to predict churn probability.")

st.info("This is a machine learning model prediction and may not be 100% accurate.")


st.header("Customer Information")

if 'total_charges' not in st.session_state:
    st.session_state.total_charges = 0.0

gender = st.selectbox("Gender", ['Male', 'Female'])

senior = st.selectbox('Senior Citizen', ['Yes', 'No'])
senior = 1 if senior=='Yes' else 0

partner = st.selectbox('Has Partner?', ['Yes', 'No'])

dependents = st.selectbox('Has Dependents?', ['Yes', 'No'])

internet = st.selectbox('Internet Service', ['DSL', 'Fiber Optic', 'No'])

payment = st.selectbox('Payment Method', 
                        [
                            'Electronic Check', 
                            'Mailed Check', 
                            'Bank transfer (automatic)', 
                            'Credit Card (automatic)'
                        ])

tenure = st.number_input("Tenure (months)", min_value=0)

monthly_charges = st.number_input("Monthly Charges")

st.session_state.total_charges = tenure * monthly_charges

total_charges = st.number_input("Total Charges", value=st.session_state.total_charges)
st.caption('Total Charges is usually Monthly Charges x Tenure')

contract = st.selectbox(
    "Contract Type",
    ['Month-to-Month', 'One year', 'Two year']
)

input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "PaperlessBilling": "Yes",
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment
    }

if st.button('Predict'):

    df = pd.DataFrame([input_data])

    required_cols = joblib.load(MODEL_DIR/"model_columns.pkl")

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    y_pred, y_prob = predict(df)

    prediction = 'Yes' if y_pred[0] == 1 else 'No'
    prob = round(y_prob[0], 3)

    st.subheader('Prediction Result')

    if prediction == 'Yes':
        st.error('Customer is likely to churn')
    else:
        st.success('Customer is likely to stay')
    
    st.write(f'**Churn Probability :** {prob:.2f}')