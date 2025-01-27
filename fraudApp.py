import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import random
import plotly.express as px
import os
from datetime import datetime

# Streamlit UI (Ensure set_page_config is the first Streamlit command)
st.set_page_config(page_title="Real-Time Fraud Detection Dashboard", layout="wide")

# Load the trained model
model = joblib.load("random_forest.pkl")  # Change to any other model you want to use

# Check if scaler file exists
scaler_path = "scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.warning("⚠️ Warning: `scaler.pkl` not found. Using default settings.")
    scaler = None  # Allow model to work without scaling

# Function to generate realistic IP addresses
def generate_ip():
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"

# Function to generate synthetic transaction data
def generate_transaction():
    user_avg_amount = np.random.uniform(1000, 50000)
    transaction_amount = np.random.uniform(500, 100000)
    user_home_city = np.random.choice(["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"])
    transaction_city = np.random.choice(["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"])
    merchant_category = np.random.choice(["Groceries", "Electronics", "Travel", "Dining", "Healthcare", "Clothing"])
    device_type = np.random.choice(["Mobile", "Desktop"])
    ip_address = generate_ip()
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

st.title("💳 Real-Time Credit Card Fraud Detection Dashboard")

st.sidebar.header("⚙️ Settings")
simulation_speed = st.sidebar.slider("Transaction Generation Speed (seconds)", 1, 10, 3)

data_placeholder = st.empty()

total_transactions = 0
fraudulent_transactions = 0
transactions = []

col1, col2 = st.columns(2)

while True:
    transaction = generate_transaction()
    transactions.append(transaction)
    total_transactions += 1
    
    # Convert transaction to DataFrame for prediction
    df = pd.DataFrame([transaction])
    df = df.drop(columns=["timestamp"])  # Timestamp is not needed for model prediction
    
    # Preprocess categorical data (ensure encoding matches training)
    categorical_cols = ['user_home_city', 'transaction_city', 'merchant_category', 'device_type']
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes  # Convert to category codes
    
    # Ensure numeric columns are in the correct format
    df = df.astype(float)
    
    # Scale input data if scaler is available
    if scaler:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df  # Use raw data if scaler is missing
    
    # Make prediction
    prediction = model.predict(df_scaled)[0]
    is_fraud = "Yes" if prediction == 1 else "No"
    transaction["fraudulent"] = is_fraud
    if prediction == 1:
        fraudulent_transactions += 1
    
    # Display transactions
    transactions_df = pd.DataFrame(transactions)
    data_placeholder.dataframe(transactions_df)
    
    with col1:
        st.metric("📊 Total Transactions", total_transactions)
    with col2:
        st.metric("⚠️ Fraudulent Transactions", fraudulent_transactions)
    
    # Visualization
    st.subheader("📈 Transaction Trends")
    if len(transactions) > 10:
        fig = px.line(transactions_df, x="timestamp", y="transaction_amount", color="fraudulent", title="Transaction Amount Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🌍 Transaction Locations")
    city_counts = transactions_df["transaction_city"].value_counts().reset_index()
    city_counts.columns = ["City", "Count"]
    fig2 = px.bar(city_counts, x="City", y="Count", title="Number of Transactions per City", color="City")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("📡 Fraud Distribution by Device Type")
    fraud_device_counts = transactions_df.groupby("device_type")["fraudulent"].value_counts().unstack().fillna(0)
    fig3 = px.bar(fraud_device_counts, barmode="stack", title="Fraud Cases by Device Type")
    st.plotly_chart(fig3, use_container_width=True)
    
    time.sleep(simulation_speed)
