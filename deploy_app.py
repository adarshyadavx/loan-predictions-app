import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="FinAI - Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(model_path="loan_model.pkl"):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

st.title("FinAI Loan Approval Portal")
st.caption("Instant Risk Evaluation & Decision Engine powered by Machine Learning")
st.write("---")

model = load_model()

if model is None:
    st.error("Trained model file loan_model.pkl not found! Please run the training script first (python3 train_model.py) to generate the model.")
else:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Applicant Details")
        st.write("Adjust the parameters below to evaluate the loan application:")
        
        age = st.slider("Applicant Age (Years)", min_value=18, max_value=150, value=35, step=1,
                        help="Typically between 18 and 80. Values above 100 will be automatically capped by the preprocessor.")
        
        income = st.number_input("Annual Income ($)", min_value=5000.0, max_value=5000000.0, value=75000.0, step=5000.0,
                                  format="%.2f", help="Annual gross income. Extremely high values will be capped during model inference.")
        
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000.0, max_value=2000000.0, value=150000.0, step=5000.0,
                                     format="%.2f", help="The total principal amount requested by the applicant.")
        
        credit_score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=680, step=1,
                                 help="Standard credit score range between 300 and 850.")
        
        dti = loan_amount / income
        
        info_lines = [
            f"Calculated Debt-to-Income (DTI) Ratio: {dti:.4f} (Loan Amount / Income)",
            f"Age Status: Capped to 80" if age > 100 else "Age Status: Normal",
            f"Income Status: Capped at fitted threshold" if income > 1500000 else "Income Status: Normal"
        ]
        st.info(" | ".join(info_lines))
        
    with col2:
        st.subheader("Decision Results")
        st.write("Model prediction output and probability scores:")
        
        input_data = pd.DataFrame([{
            'Age': age,
            'Income': income,
            'LoanAmount': loan_amount,
            'CreditScore': credit_score
        }])
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        prob_approved = probabilities[1]
        
        if prediction == 1:
            st.success("Loan Approved")
            st.write("The applicant meets the eligibility criteria and presents a low credit risk profile.")
        else:
            st.error("Loan Rejected")
            st.write("The applicant does not meet the eligibility requirements or presents a high default risk.")
            
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(label="Approval Probability", value=f"{prob_approved:.1%}")
        with m_col2:
            st.metric(label="Credit Score", value=str(credit_score))
        with m_col3:
            st.metric(label="DTI Ratio", value=f"{dti:.2f}")
            
        st.write("---")
        
        st.write("Key Drivers of the Approval Decision:")
        st.write(
            "Our model evaluates the approval based on 5 features. The most critical features in the Decision Tree are Debt-to-Income Ratio (DTI) and Credit Score."
        )
        
        with st.expander("View Raw Model Input Data Structure"):
            st.dataframe(input_data)
