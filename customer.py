import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.compose import ColumnTransformer
import streamlit as st
import pickle

st.title("Customer Churn Prediction using ML")
sample = []
result = None
gender = st.radio('Enter your Gender',['Male', 'Female'])
sample.append(gender)
SeniorCitizen = st.radio('Are You a Senior Citizen?',[1, 0],captions=['Yes','No'])
sample.append(SeniorCitizen)
Partner = st.radio('Do you have a Partner?',['Yes', 'No']) 
sample.append(Partner)
Dependents = st.radio('Do you have Dependents?',['Yes', 'No'])
sample.append(Dependents)
tenure = st.number_input("Enter tenure")
sample.append(tenure)
PhoneService = st.radio('Do you have a mobile service?',['Yes', 'No'])
sample.append(PhoneService)
MultipleLines = st.selectbox('Do you have multiple lines',('No phone service','Yes', 'No'))
sample.append(MultipleLines)
InternetService = st.selectbox('Type of Internet Service',('DSL', 'Fiber optic', 'No'))
sample.append(InternetService)
OnlineSecurity = st.selectbox('Do you have online security?',('Yes','No', 'No internet service'))
sample.append(OnlineSecurity)
OnlineBackup = st.selectbox('Do you have online backup?',('Yes','No','No internet service'))
sample.append(OnlineBackup)
DeviceProtection = st.selectbox('Do you have device protection?',('Yes','No','No internet service'))
sample.append(DeviceProtection)
TechSupport = st.selectbox('Do you have tech support?',('Yes','No','No internet service'))
sample.append(TechSupport)
StreamingTV = st.selectbox('Do you have a streaming TV?',('Yes','No','No internet service'))
sample.append(StreamingTV)
StreamingMovies = st.selectbox('Do you have a streaming movies?',('Yes','No','No internet service'))
sample.append(StreamingMovies)
Contract = st.selectbox('Type of Contract',('Month-to-month', 'One year', 'Two year'))
sample.append(Contract)
PaperlessBilling = st.radio('Do you have paperless billing?',['Yes', 'No'])
sample.append(PaperlessBilling)
PaymentMethod = st.selectbox('Method of Payment',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
sample.append(PaymentMethod)
MonthlyCharges = st.number_input("Enter monthly charges")
sample.append(MonthlyCharges)
TotalCharges = st.number_input("Enter total charges")
sample.append(TotalCharges)

import os
# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Construct the path to the pickle file relative to the current script
pickle_file_path = os.path.join(current_dir, "churn_model.pkl")
with open(pickle_file_path, "rb") as f:
    model = pickle.load(f)
if st.button("Submit") == True:
    result_df = pd.DataFrame(np.array([sample]).reshape(1,-1),columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges'])
    result = model.predict(result_df)[0]
else:
    pass

if result == "Yes":
    st.subheader("The customer churned")
    st.image("https://www.google.com/imgres?q=customer%20churn%20prediction&imgurl=https%3A%2F%2Fdaxg39y63pxwu.cloudfront.net%2Fimages%2Fblog%2Fchurn-models%2FCustomer_Churn_Prediction_Models_in_Machine_Learning.png&imgrefurl=https%3A%2F%2Fwww.projectpro.io%2Farticle%2Fchurn-models%2F709&docid=n8tJT-aNut6L1M&tbnid=3j1GQSNNX6bllM&vet=12ahUKEwi634CUgqaFAxUw3TgGHWYxCB8QM3oECBsQAA..i&w=2084&h=1250&hcb=2&ved=2ahUKEwi634CUgqaFAxUw3TgGHWYxCB8QM3oECBsQAA")
elif result == "No":
    st.subheader("Customer didn't churn]")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQuIwotrwO4bAXz25_vuwm6mf-cfxw4xqQqHA&s")

