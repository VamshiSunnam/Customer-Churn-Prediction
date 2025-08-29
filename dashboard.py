# dashboard.py
# An interactive web dashboard for customer churn prediction.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Load Model and Preprocessor ---
# Load the pre-trained neural network model and the preprocessor object.
# Use st.cache_resource to load these only once and speed up the app.
@st.cache_resource
def load_prediction_model():
    """Loads the trained Keras model."""
    try:
        model = load_model('churn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_preprocessor_object():
    """Loads the saved joblib preprocessor."""
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        return preprocessor
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None

model = load_prediction_model()
preprocessor = load_preprocessor_object()

# --- Dashboard Interface ---
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title and description
st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("""
This dashboard uses a trained neural network to predict whether a customer is likely to churn. 
Enter the customer's details in the sidebar to get a prediction.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Customer Details")

def get_user_input():
    """Creates sidebar widgets to collect customer data."""
    gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
    age = st.sidebar.slider("Age", 18, 100, 35)
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
    contract = st.sidebar.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
    tech_support = st.sidebar.selectbox("Has Tech Support?", ('Yes', 'No'))
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 50.0, 0.5)
    
    # Calculate a plausible TotalCharges
    total_charges = tenure * monthly_charges

    # Create a dictionary of the input data
    input_data = {
        'Gender': gender,
        'Age': age,
        'TenureMonths': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'HasTechSupport': tech_support
    }
    
    # Convert to a DataFrame
    features = pd.DataFrame(input_data, index=[0])
    return features


# Get input from the user
user_input_df = get_user_input()

# Display the user's input
st.subheader("Customer Input Summary")
st.write(user_input_df)

# --- Prediction Logic ---
if st.sidebar.button("Predict Churn"):
    if model is not None and preprocessor is not None:
        try:
            # Preprocess the user input
            input_processed = preprocessor.transform(user_input_df)
            
            # Make a prediction
            prediction_proba = model.predict(input_processed)[0][0]
            prediction = 1 if prediction_proba > 0.5 else 0

            # --- Display Prediction Result ---
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"**Prediction: Churn** (Probability: {prediction_proba:.2%})")
                st.warning("This customer is at a high risk of churning. Consider taking retention actions.")
            else:
                st.success(f"**Prediction: No Churn** (Probability of Churn: {prediction_proba:.2%})")
                st.info("This customer is likely to stay.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model or preprocessor could not be loaded. Please check the files.")

