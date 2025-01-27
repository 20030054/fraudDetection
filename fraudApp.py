import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# Load the trained model
model = joblib.load("random_forest.pkl")  # Change to any other model you want to use

# Function to generate synthetic transaction data
def generate_transaction():
    user_avg_amount = np.random.uniform(1000, 50000)
    transaction_amount = np.random.uniform(500, 100000)
    user_home_city = np.random.choice(["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"])
    transaction_city = np.random.choice(["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"])
    merchant_category = np.random.choice(["Groceries", "Electronics", "Travel", "Dining", "Healthcare", "Clothing"])
    device_type = np.random.choice(["Mobile", "Desktop"])
    ip_address = hash(str(np.random.randint(1, 255))) % 10**6
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "user_avg_amount": user_avg_amount,
        "transaction_amount": transaction_amount,
        "user_home_city": user_home_city,
        "transaction_city": transaction_city,
        "merchant_category": merchant_category,
        "device_type": device_type,
        "ip_address": ip_address,
        "timestamp": timestamp
    }

# Streamlit UI
st.title("Real-Time Credit Card Fraud Detection")

st.sidebar.header("Settings")
simulation_speed = st.sidebar.slider("Transaction Generation Speed (seconds)", 1, 10, 3)

data_placeholder = st.empty()

transactions = []

while True:
    transaction = generate_transaction()
    transactions.append(transaction)
    
    # Convert transaction to DataFrame for prediction
    df = pd.DataFrame([transaction])
    df = df.drop(columns=["timestamp"])  # Timestamp is not needed for model prediction
    
    # Preprocess categorical data (same as in training script)
    categorical_cols = ['user_home_city', 'transaction_city', 'merchant_category', 'device_type']
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes
    
    # Make prediction
    prediction = model.predict(df)[0]
    transaction["fraudulent"] = "Yes" if prediction == 1 else "No"
    
    # Display transactions
    transactions_df = pd.DataFrame(transactions)
    data_placeholder.dataframe(transactions_df)
    
    time.sleep(simulation_speed)
