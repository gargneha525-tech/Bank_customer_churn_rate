import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    layout="centered"
)


# =========================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# =========================================================

@st.cache_resource
def load_model_and_preprocessors():

    model = tf.keras.models.load_model("final_bank_churn_ann_model.keras")

    geography_encoder = joblib.load("geography_encoder.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")
    scaler = joblib.load("feature_scaler.pkl")

    return model, geography_encoder, gender_encoder, scaler


model, geography_encoder, gender_encoder, scaler = load_model_and_preprocessors()


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "Enter the customer's details below to predict whether "
    "the customer is likely to churn."
)

st.divider()


# =========================================================
# CUSTOMER INPUT
# =========================================================

st.subheader("Customer Information")


credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650,
    step=1
)


geography = st.selectbox(
    "Geography",
    options=list(geography_encoder.classes_)
)


gender = st.selectbox(
    "Gender",
    options=list(gender_encoder.classes_)
)


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40,
    step=1
)


tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5,
    step=1
)


balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=60000.0,
    step=1000.0
)


num_of_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2,
    step=1
)


has_cr_card = st.selectbox(
    "Has Credit Card?",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


is_active_member = st.selectbox(
    "Is Active Member?",
    options=[1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔮 Predict Churn", use_container_width=True):

    try:

        # -------------------------------------------------
        # Create dataframe in EXACT same feature order
        # used during model training
        # -------------------------------------------------

        new_customer = pd.DataFrame({
            "CreditScore": [credit_score],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })


        # -------------------------------------------------
        # Encode categorical variables
        # Same method used in the notebook
        # -------------------------------------------------

        new_customer["Geography"] = geography_encoder.transform(
            new_customer["Geography"]
        )

        new_customer["Gender"] = gender_encoder.transform(
            new_customer["Gender"]
        )


        # -------------------------------------------------
        # Scale features
        # Same StandardScaler used during training
        # -------------------------------------------------

        new_customer_scaled = scaler.transform(
            new_customer
        )


        # -------------------------------------------------
        # Make prediction
        # -------------------------------------------------

        churn_probability = model.predict(
            new_customer_scaled,
            verbose=0
        )[0][0]


        churn_percentage = churn_probability * 100


        # -------------------------------------------------
        # Display probability
        # -------------------------------------------------

        st.subheader("Prediction Result")

        st.metric(
            label="Churn Probability",
            value=f"{churn_percentage:.2f}%"
        )


        # -------------------------------------------------
        # Final prediction
        # Same threshold used in notebook: 0.5
        # -------------------------------------------------

        if churn_probability >= 0.5:

            st.error(
                "⚠️ Customer is likely to churn."
            )

        else:

            st.success(
                "✅ Customer is likely to stay."
            )


        # -------------------------------------------------
        # Display input data
        # -------------------------------------------------

        with st.expander("View Customer Data"):

            st.dataframe(
                new_customer,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Prediction error: {str(e)}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Bank Customer Churn Prediction using Artificial Neural Network (ANN)"
)