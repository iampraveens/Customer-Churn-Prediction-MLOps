import streamlit as st 
import pandas as pd

from src.model_dev import DecisionTreeClassifier_Model
from steps.load_data import load_df
from steps.clean_data import clean_df



df = load_df(data_path=r"C:\Users\sprav\Pictures\Customer Churn Prediction\data\telcoChurn.csv")
X_train, X_test, y_train, y_test = clean_df(df)
dt = DecisionTreeClassifier_Model()
trained_model = dt.train(X_train, y_train)


st.title("Customer Churn Prediction")

st.sidebar.header("User Input")
gender = st.sidebar.radio("Select Gender:", ["Male", "Female"])
senior_citizen = st.sidebar.checkbox("Is Senior Citizen?")
partner = st.sidebar.checkbox("Has Partner?")
dependents = st.sidebar.checkbox("Has Dependents?")
tenure = st.sidebar.number_input("Tenure (months)")
phone_service = st.sidebar.radio("Has Phone Service?", ["Yes", "No"])
multiple_lines = st.sidebar.radio("Multiple Lines?", ["Yes", "No"])
online_security = st.sidebar.radio("Online Security?", ["Yes", "No"])
online_backup = st.sidebar.radio("Online Backup?", ["Yes", "No"])
device_protection = st.sidebar.radio("Device Protection?", ["Yes", "No"])
tech_support = st.sidebar.radio("Tech Support?", ["Yes", "No"])
streaming_tv = st.sidebar.radio("Streaming TV?", ["Yes", "No"])
streaming_movies = st.sidebar.radio("Streaming Movies?", ["Yes", "No"])
paperless_billing = st.sidebar.checkbox("Paperless Billing?")
monthly_charges = st.sidebar.number_input("Monthly Charges:")
total_charges = st.sidebar.number_input("Total Charges:")
internet_fiber_optic = st.sidebar.radio("InternetService_Fiber optic", ["Yes", "No"])
internet_no = st.sidebar.radio("InternetService_No", ["Yes", "No"])
contract_one_year = st.sidebar.radio("Contract_One year", ["Yes", "No"])
contract_two_year = st.sidebar.radio("Contract_Two year", ["Yes", "No"])
payment_credit_card = st.sidebar.radio("PaymentMethod_Credit card (automatic)", ["Yes", "No"])
payment_electronic_check = st.sidebar.radio("PaymentMethod_Electronic check", ["Yes", "No"])
payment_mailed_check = st.sidebar.radio("PaymentMethod_Mailed check", ["Yes", "No"])
# Create a feature vector from the user input
user_input = pd.DataFrame({
    "gender": [1 if gender == "Female" else 0],
    "SeniorCitizen": [1 if senior_citizen else 0],
    "Partner": [1 if partner else 0],
    "Dependents": [1 if dependents else 0],
    "tenure": [tenure],
    "PhoneService": [1 if phone_service == "Yes" else 0],
    "MultipleLines": [1 if multiple_lines == "Yes" else 0],
    "OnlineSecurity": [1 if online_security == "Yes" else 0],
    "OnlineBackup": [1 if online_backup == "Yes" else 0],
    "DeviceProtection": [1 if device_protection == "Yes" else 0],
    "TechSupport": [1 if tech_support == "Yes" else 0],
    "StreamingTV": [1 if streaming_tv == "Yes" else 0],
    "StreamingMovies": [1 if streaming_movies == "Yes" else 0],
    "PaperlessBilling": [1 if paperless_billing else 0],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "InternetService_Fiber optic": [1 if internet_fiber_optic == "Yes" else 0],
    "InternetService_No": [1 if internet_no == "Yes" else 0],
    "Contract_One year": [1 if contract_one_year == "Yes" else 0],
    "Contract_Two year": [1 if contract_two_year == "Yes" else 0],
    "PaymentMethod_Credit card (automatic)": [1 if payment_credit_card == "Yes" else 0],
    "PaymentMethod_Electronic check": [1 if payment_electronic_check == "Yes" else 0],
    "PaymentMethod_Mailed check": [1 if payment_mailed_check == "Yes" else 0]
})

# Make predictions using the trained model
if st.button("Predict"):
    prediction = trained_model.predict(user_input)

    # Display the prediction result
    st.write("Churn Prediction Result:")
    if prediction[0] == 0:
        st.write("Customer is likely to stay.")
    else:
        st.write("Customer is likely to churn.")

# You can also add more elements to display additional information or visualizations.
