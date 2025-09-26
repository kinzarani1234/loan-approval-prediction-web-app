import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Load Trained Models & Objects

log_reg = joblib.load("log_reg_model.pkl")   # Logistic Regression
rf = joblib.load("rf_model.pkl")             # Random Forest
scaler = joblib.load("scaler.pkl")           # Scaler (for Logistic Regression)
train_columns = joblib.load("train_columns.pkl")  # Feature order


# App UI

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰", layout="wide")

st.title(" Loan Approval Prediction App")
st.write("Fill the form below to predict loan approval using **Logistic Regression** or **Random Forest**.")


# Sidebar Instructions

st.sidebar.header("ℹ️ How it Works")
st.sidebar.markdown("""
1. Provide values for all loan application features.  
2. For **categorical features**, use the dropdown.  
3. For **numerical features**, use the sliders.  
4. Choose the ML model (Logistic Regression or Random Forest).  
5. Click **Predict** to get the result.  
""")

# ==========================
# Features
# ==========================
st.subheader(" Input Features")

input_data = {}

#  Categorical Inputs
col1, col2, col3 = st.columns(3)
with col1:
    input_data["education"] = st.selectbox("Education", ["Graduate", "Not Graduate"])
with col2:
    input_data["self_employed"] = st.selectbox("Self Employed", ["Yes", "No"])
with col3:
    input_data["no_of_dependents"] = st.slider("No. of Dependents", 0, 10, 0)

#  Numerical Inputs
col4, col5, col6 = st.columns(3)
with col4:
    input_data["income_annum"] = st.slider("Annual Income", 100000, 20000000, 500000)
with col5:
    input_data["loan_amount"] = st.slider("Loan Amount", 50000, 50000000, 100000)
with col6:
    input_data["loan_term"] = st.slider("Loan Term (months)", 6, 360, 12)

col7, col8, col9 = st.columns(3)
with col7:
    input_data["cibil_score"] = st.slider("CIBIL Score", 300, 900, 650)
with col8:
    input_data["residential_assets_value"] = st.slider("Residential Assets Value", 0, 50000000, 1000000)
with col9:
    input_data["commercial_assets_value"] = st.slider("Commercial Assets Value", 0, 50000000, 1000000)

col10, col11 = st.columns(2)
with col10:
    input_data["luxury_assets_value"] = st.slider("Luxury Assets Value", 0, 50000000, 500000)
with col11:
    input_data["bank_asset_value"] = st.slider("Bank Asset Value", 0, 50000000, 500000)


input_df = pd.DataFrame([input_data])

# Encode categorical features same as training
input_df["education"] = input_df["education"].map({"Not Graduate": 0, "Graduate": 1})
input_df["self_employed"] = input_df["self_employed"].map({"No": 0, "Yes": 1})

# Align columns with training
input_encoded = input_df.reindex(columns=train_columns, fill_value=0)


st.markdown("###  Choose Model")
model_choice = st.radio("Select Model:", ("Logistic Regression", "Random Forest"))


if st.button("Predict"):
    if model_choice == "Logistic Regression":
        X_scaled = scaler.transform(input_encoded)
        prediction = log_reg.predict(X_scaled)[0]
    else:
        prediction = rf.predict(input_encoded)[0]

    if prediction == 1:
        st.success(" Loan will be **Approved**")
    else:
        st.error(" Loan will be **Rejected**")
