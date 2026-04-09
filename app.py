import streamlit as st
import pandas as pd
import requests

from src.config import API_URL

# Main Information
st.title("📊 Customer Churn Prediction")

st.markdown("Enter customer details to predict churn probability.")

st.info("This is a machine learning model prediction and may not be 100% accurate.")

# Initializing Total Charges field to 0.0
if 'total_charges' not in st.session_state:
    st.session_state.total_charges = 0.0


# Customer Info Section (taking input from user)
st.header("Customer Information")

gender = st.selectbox("Gender", ['Male', 'Female'])

senior = st.selectbox('Senior Citizen', ['Yes', 'No'])
senior = 1 if senior=='Yes' else 0

partner = st.selectbox('Has Partner?', ['Yes', 'No'])

dependents = st.selectbox('Has Dependents?', ['Yes', 'No'])

internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

payment = st.selectbox('Payment Method', 
                        [
                            'Electronic check', 
                            'Mailed check', 
                            'Bank transfer (automatic)', 
                            'Credit card (automatic)'
                        ])

tenure = st.number_input("Tenure (months)", min_value=0)

monthly_charges = st.number_input("Monthly Charges [USD]")

# Making Total Charges field dyanmically equal to tenure * monthly_charges
st.session_state.total_charges = tenure * monthly_charges

total_charges = st.number_input("Total Charges [USD]", value=st.session_state.total_charges)
st.caption('Total Charges is usually Monthly Charges x Tenure')


contract = st.selectbox(
    "Contract Type",
    ['Month-to-month', 'One year', 'Two year']
)

options = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'Contract',
    'InternetService',
    'PaymentMethod'
]

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


# On pressing Predict Button
if st.button('Predict'):

    with st.spinner('Predicting...'):
        response = requests.post(API_URL + "/predict", json=input_data)

    if response.status_code == 200:
        result = response.json()

        prediction = result['prediction']
        probability = result['probability']
        factors = pd.DataFrame(result['factors'])
    
    else:
        st.error('API request failed')
        st.stop()


    # y_pred, y_prob = predict(df)

    st.subheader('Prediction Result')

    if prediction == 'Yes':
        st.error('Customer is likely to churn')
    else:
        st.success('Customer is likely to stay')
    
    st.write(f'**Churn Probability :** {probability}%')

    if not factors.empty:
        if prediction == 'Yes':
            st.subheader('Why is this customer likely to churn?')
            for _, row in factors.iterrows():
                st.write(f"{row['Feature'].split('_')[0]} increased the likelihood of churn")
        else:
            st.subheader('Why is this customer likely to stay?')
            for _, row in factors.iterrows():
                st.write(f"{row['Feature'].split('_')[0]} decreased the likelihood of churn")