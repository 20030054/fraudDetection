import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import os

# Load the pre-trained Random Forest model
MODEL_PATH = "random_forest.pkl"
ENCODERS_PATH = "label_encoders.pkl"

# Ensure model and encoders exist before loading
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
else:
    st.error("🚨 Model or label encoders file not found! Please check the file paths.")
    st.stop()

# Function to generate synthetic transactions
def generate_synthetic_transaction():
    """Generates a synthetic transaction for real-time testing."""
    cities = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad', 'Peshawar', 'Quetta']
    merchant_categories = ['Electronics', 'Travel', 'Jewelry', 'Online Services', 'Groceries', 'Retail', 'Utilities', 'Healthcare']
    devices = ['Mobile', 'Desktop']
    
    transaction = {
        'user_id': np.random.randint(1000, 9999),
        'user_avg_amount': np.random.normal(5000, 1500),
        'transaction_amount': np.random.uniform(100, 50000),
        'user_home_city': np.random.choice(cities),
        'transaction_city': np.random.choice(cities),
        'merchant_category': np.random.choice(merchant_categories),
        'device_type': np.random.choice(devices),
        'ip_address': f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
    }
    return transaction

# Function to preprocess the transaction for the model
def preprocess_transaction(transaction):
    """Preprocesses the transaction data for the model."""
    try:
        # Encode categorical features
        transaction['user_home_city'] = label_encoders['user_home_city'].transform([transaction['user_home_city']])[0]
        transaction['transaction_city'] = label_encoders['transaction_city'].transform([transaction['transaction_city']])[0]
        transaction['merchant_category'] = label_encoders['merchant_category'].transform([transaction['merchant_category']])[0]
        transaction['device_type'] = label_encoders['device_type'].transform([transaction['device_type']])[0]
        
        # Hash IP address
        transaction['ip_address'] = hash(transaction['ip_address']) % 10**6
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        return df[['user_avg_amount', 'transaction_amount', 'user_home_city', 'transaction_city', 'merchant_category', 'device_type', 'ip_address']]
    
    except Exception as e:
        st.error(f"⚠️ Error in preprocessing transaction: {e}")
        return None

# Streamlit app title
st.title("💳 Real-Time Fraud Detection Dashboard")
st.write("This dashboard simulates real-time credit card transactions and detects fraud using a pre-trained Random Forest model.")

# Initialize session state for storing transactions
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=[
        'user_id', 'user_avg_amount', 'transaction_amount', 'user_home_city', 
        'transaction_city', 'merchant_category', 'device_type', 'ip_address', 'is_fraud'
    ])

# Sidebar for user controls
st.sidebar.header("⚙️ Controls")
update_interval = st.sidebar.slider("⏳ Update Interval (seconds)", 1, 10, 2)
num_transactions = st.sidebar.slider("📊 Number of Transactions to Display", 10, 100, 20)

# Real-time transaction feed
st.header("📌 Real-Time Transaction Feed")
placeholder = st.empty()

# Fraud distribution chart
st.header("🔍 Fraud Distribution")
fraud_chart_placeholder = st.empty()

# Transaction trends chart
st.header("📈 Transaction Trends")
trends_chart_placeholder = st.empty()

# Simulate real-time transactions
while True:
    # Generate a synthetic transaction
    transaction = generate_synthetic_transaction()
    
    # Preprocess the transaction
    processed_transaction = preprocess_transaction(transaction)
    
    if processed_transaction is not None:
        # Predict fraud
        is_fraud = model.predict(processed_transaction)[0]

        # Reduce fraud occurrences to ~10%
        if is_fraud == 1:
            is_fraud = np.random.choice([0, 1], p=[0.9, 0.1])  # 90% legitimate, 10% fraud
        
        transaction['is_fraud'] = is_fraud

        # Add transaction to session state
        st.session_state.transactions = pd.concat([st.session_state.transactions, pd.DataFrame([transaction])], ignore_index=True)

        # Keep only the latest N transactions
        if len(st.session_state.transactions) > num_transactions:
            st.session_state.transactions = st.session_state.transactions.iloc[-num_transactions:]

        # Display real-time transaction feed
        with placeholder.container():
            st.write("### 🔄 Latest Transactions")
            st.dataframe(st.session_state.transactions.tail(10))

        # Update fraud distribution chart
        with fraud_chart_placeholder.container():
            st.write("### 📊 Fraud Distribution")

            if not st.session_state.transactions.empty:
                fraud_counts = st.session_state.transactions['is_fraud'].value_counts()

                # Dynamically generate labels based on available fraud categories
                labels = fraud_counts.index.map(lambda x: "Legitimate" if x == 0 else "Fraud").tolist()

                # Define colors dynamically based on available labels
                colors = ['green' if label == "Legitimate" else 'red' for label in labels]

                fig, ax = plt.subplots()
                ax.pie(fraud_counts, labels=labels, autopct='%1.1f%%', colors=colors)
                st.pyplot(fig)
            else:
                st.write("No transactions available for fraud distribution.")

        # Update transaction trends chart
        with trends_chart_placeholder.container():
            st.write("### 📉 Transaction Trends")

            if not st.session_state.transactions.empty:
                fig, ax = plt.subplots()
                ax.plot(st.session_state.transactions['transaction_amount'], label='Transaction Amount', marker='o')
                ax.set_xlabel("Transaction Index")
                ax.set_ylabel("Amount (PKR)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.write("No transactions available for trend analysis.")

    # Wait for the specified interval before generating the next transaction
    time.sleep(update_interval)
