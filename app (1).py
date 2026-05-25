

import streamlit as st

import pandas as pd

import numpy as np

import joblib

from xgboost import XGBClassifier


# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "loan_default_xgboost.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(

    page_title="Loan Default Prediction",

    page_icon="💰",

    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .stButton>button {

        width: 100%;

        border-radius: 10px;

        height: 3em;

        background-color: #4CAF50;

        color: white;

        font-size: 18px;
    }

    </style>
    """,

    unsafe_allow_html=True
)


# =========================
# TITLE
# =========================

st.title("💳 Loan Default Prediction System")

st.write(
    "Predict whether a borrower is likely to default on a loan using Machine Learning."
)


# =========================
# SIDEBAR
# =========================

st.sidebar.header("About")

st.sidebar.write(

    """
    This application uses:

    - Logistic Regression
    - Random Forest
    - XGBoost

    Final Prediction is generated using the optimized XGBoost model.
    """
)


# =========================
# INPUT SECTION
# =========================

st.subheader("Enter Borrower Details")


col1, col2 = st.columns(2)


with col1:

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0
    )

    int_rate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        value=12.0
    )

    annual_inc = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=60000.0
    )

    dti = st.number_input(
        "Debt-To-Income Ratio",
        min_value=0.0,
        value=18.0
    )

    fico_score = st.number_input(
        "FICO Score",
        min_value=300.0,
        value=700.0
    )

    installment = st.number_input(
        "Installment",
        min_value=0.0,
        value=300.0
    )


with col2:

    revol_bal = st.number_input(
        "Revolving Balance",
        min_value=0.0,
        value=5000.0
    )

    emp_length = st.number_input(
        "Employment Length",
        min_value=0.0,
        value=5.0
    )

    open_acc = st.number_input(
        "Open Accounts",
        min_value=0.0,
        value=8.0
    )

    total_acc = st.number_input(
        "Total Accounts",
        min_value=0.0,
        value=20.0
    )

    revol_util = st.number_input(
        "Revolving Utilization",
        min_value=0.0,
        value=45.0
    )

    inq_last_6mths = st.number_input(
        "Inquiries Last 6 Months",
        min_value=0.0,
        value=1.0
    )


# =========================
# FEATURE ENGINEERING
# =========================

income_to_loan = annual_inc / (loan_amnt + 1)

installment_ratio = installment / (annual_inc + 1)

risk_interaction = dti * int_rate

credit_load = revol_bal / (annual_inc + 1)


# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame({

    'loan_amnt': [loan_amnt],

    'int_rate': [int_rate],

    'annual_inc': [annual_inc],

    'dti': [dti],

    'fico_score': [fico_score],

    'installment': [installment],

    'revol_bal': [revol_bal],

    'emp_length': [emp_length],

    'open_acc': [open_acc],

    'total_acc': [total_acc],

    'revol_util': [revol_util],

    'inq_last_6mths': [inq_last_6mths],

    'income_to_loan': [income_to_loan],

    'installment_ratio': [installment_ratio],

    'risk_interaction': [risk_interaction],

    'credit_load': [credit_load]
})


# =========================
# HANDLE MISSING COLUMNS
# =========================

for col in feature_columns:

    if col not in input_data.columns:

        input_data[col] = 0


input_data = input_data[
    feature_columns
]


# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Loan Risk"):

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]


    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ High Risk Borrower - Likely To Default"
        )

    else:

        st.success(
            "✅ Low Risk Borrower - Loan Likely Safe"
        )


    st.metric(

        label="Default Probability",

        value=f"{probability:.2%}"
    )


    # =====================
    # RISK LEVEL
    # =====================

    if probability < 0.30:

        st.success("Risk Level : LOW")

    elif probability < 0.60:

        st.warning("Risk Level : MEDIUM")

    else:

        st.error("Risk Level : HIGH")


    # =====================
    # PROBABILITY BAR
    # =====================

    st.progress(float(probability))


