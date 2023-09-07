import streamlit as st 
import pandas as pd
import numpy as np
from PIL import Image
import sys

from ..src.utils import LoadModel
from pipelines.prediction_pipeline import predict_model 

sys.path.append("..")
sys.path.append("../src/")

image = Image.open(r'C:\Users\sprav\Pictures\Customer Churn Prediction\assets\telco.png')
page_title = 'Telcom Churn Prediction'
page_icon = image
layout = 'wide'

st.set_page_config(page_title=page_title,
                   page_icon=page_icon,
                   layout=layout
                   )
hide_style = '''
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <style>
            
            '''
header_style = '''
             <style>
             .navbar {
                 position: fixed;
                 top: 0;
                 left: 0;
                 width: 100%;
                 z-index: 1;
                 display: flex;
                 justify-content: center;
                 align-items: center;
                 height: 80px;
                 background-color: #174189;
                 box-sizing: border-box;
             }
             
             .navbar-brand {
                 color: white !important;
                 font-size: 23px;
                 text-decoration: none;
                 margin-right: auto;
                 margin-left: 50px;
             }
             
             .navbar-brand img {
                margin-bottom: 6px;
                margin-right: 1px;
                width: 40px;
                height: 40px;
                justify-content: center;
            }
            
            /* Add the following CSS to change the color of the text */
            .navbar-brand span {
                color: #9973d5;
                justify-content: center;
            }
            
             </style>
             
             <nav class="navbar">
                 <div class="navbar-brand">
                <img src="https://cdn-icons-png.flaticon.com/512/9815/9815472.png" alt="Logo">
                    Customer Churn Prediction
                 </div>
             </nav>
               '''
st.markdown(hide_style, unsafe_allow_html=True)
st.markdown(header_style, unsafe_allow_html=True)

column_1, column_2, column_3, column_4 = st.columns([20,20,20,25])

with column_1:
    Partner = st.selectbox('Parter', ['Yes', 'No'])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    InternetService_Fiber_optic = st.selectbox("InternetService Fiber optic", ["Yes", "No"])
    PaymentMethod_Credit_card = st.selectbox("PaymentMethod Credit card (automatic)", ["Yes", "No"])
    
    predict_button = st.button("Predict", type='primary')

with column_2:
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
    MonthlyCharges = st.number_input("Monthly Charges")
    Contract_One_year = st.selectbox("Contract One year", ["Yes", "No"])
    PaymentMethod_Electronic_check = st.selectbox("PaymentMethod Electronic check", ["Yes", "No"])

with column_3:
    tenure = st.number_input("Tenure (months)")
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
    TotalCharges = st.number_input("Total Charges")
    Contract_Two_year = st.selectbox("Contract Two year", ["Yes", "No"])
    PaymentMethod_Mailed_check = st.selectbox("PaymentMethod Mailed check", ["Yes", "No"])
    
user_input = pd.DataFrame({
    "Partner": [1 if Partner == 'Yes' else 0],
    "Dependents": [1 if Dependents == 'Yes' else 0],
    "tenure": [tenure],
    "OnlineSecurity": [1 if OnlineSecurity == "Yes" else 0],
    "OnlineBackup": [1 if OnlineBackup == "Yes" else 0],
    "DeviceProtection": [1 if DeviceProtection == "Yes" else 0],
    "TechSupport": [1 if TechSupport == "Yes" else 0],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges],
    "InternetService_Fiber optic": [1 if InternetService_Fiber_optic == "Yes" else 0],
    "Contract_One year": [1 if Contract_One_year == "Yes" else 0],
    "Contract_Two year": [1 if Contract_Two_year == "Yes" else 0],
    "PaymentMethod_Credit card (automatic)": [1 if PaymentMethod_Credit_card == "Yes" else 0],
    "PaymentMethod_Electronic check": [1 if PaymentMethod_Electronic_check == "Yes" else 0],
    "PaymentMethod_Mailed check": [1 if PaymentMethod_Mailed_check == "Yes" else 0]
})
    
load_path = r'C:\Users\sprav\Pictures\Customer Churn Prediction\saved_models\model.pkl'
loaded_model = LoadModel(load_path=load_path)

with column_1:
    if predict_button:
        prediction = loaded_model.predict(user_input)

        # Display the prediction result
        if prediction[0] == 0:
            colored1 = f'<span style="color: #174189;font-size: 25px; font-weight: bold;">Customer is likely to stay.</span>'
            st.markdown(colored1, unsafe_allow_html=True)
        else:
            colored2 = f'<span style="color: #174189;font-size: 25px; font-weight: bold;">Customer is likely to churn.</span>'
            st.markdown(colored2, unsafe_allow_html=True)
        
