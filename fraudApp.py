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

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
else:
    st.error("🚨 Model or label encoders file not found! Please check the file paths.")
    st.stop()

# Function to generate synthetic transactions
def generate_synthetic_transaction():
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
    try:
        transaction['user_home_city'] = label_encoders['user_home_city'].transform([transaction['user_home_city']])[0]
        transaction['transaction_city'] = label_encoders['transaction_city'].transform([transaction['transaction_city']])[0]
        transaction['merchant_category'] = label_encoders['merchant_category'].transform([transaction['merchant_category']])[0]
        transaction['device_type'] = label_encoders['device_type'].transform([transaction['device_type']])[0]
        
        transaction['ip_address'] = hash(transaction['ip_address']) % 10**6
        
        df = pd.DataFrame([transaction])
        return df[['user_avg_amount', 'transaction_amount', 'user_home_city', 'transaction_city', 'merchant_category', 'device_type', 'ip_address']]
    
    except Exception as e:
        st.error(f"⚠️ Error in preprocessing transaction: {e}")
        return None

# Streamlit app title
st.title("💳 Real-Time Fraud Detection Dashboard")

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = pd.DataFrame(columns=[
        'user_id', 'user_avg_amount', 'transaction_amount', 'user_home_city', 
        'transaction_city', 'merchant_category', 'device_type', 'ip_address', 'is_fraud'
    ])

if 'fraud_alerts' not in st.session_state:
    st.session_state.fraud_alerts = []  # Stores last 5 fraud alerts

# Sidebar controls
st.sidebar.header("⚙️ Controls")
update_interval = st.sidebar.slider("⏳ Update Interval (seconds)", 1, 10, 2)
num_transactions = st.sidebar.slider("📊 Number of Transactions to Display", 10, 100, 20)

# 🚨 Fraud Alerts Section
st.sidebar.header("🚨 Recent Fraud Alerts")
fraud_alert_placeholder = st.sidebar.empty()

# Real-time transaction feed
st.header("📌 Real-Time Transaction Feed")
placeholder = st.empty()

# Fraud distribution chart
st.header("🔍 Fraud Distribution")
fraud_chart_placeholder = st.empty()

# Transaction trends chart
st.header("📈 Transaction Trends")
trends_chart_placeholder = st.empty()

# Additional Insights
st.header("📊 Additional Insights")
fraud_trend_placeholder = st.empty()
category_distribution_placeholder = st.empty()
amount_comparison_placeholder = st.empty()

# Simulate real-time transactions
while True:
    transaction = generate_synthetic_transaction()
    processed_transaction = preprocess_transaction(transaction)
    
    if processed_transaction is not None:
        is_fraud = model.predict(processed_transaction)[0]
        if is_fraud == 1:
            is_fraud = np.random.choice([0, 1], p=[0.9, 0.1])  

        transaction['is_fraud'] = is_fraud
        st.session_state.transactions = pd.concat([st.session_state.transactions, pd.DataFrame([transaction])], ignore_index=True)

        if len(st.session_state.transactions) > num_transactions:
            st.session_state.transactions = st.session_state.transactions.iloc[-num_transactions:]

        # 🚨 Show alerts for fraudulent transactions
        if is_fraud == 1:
            alert_msg = f"🚨 Fraud Alert! User {transaction['user_id']} | Amount: PKR {transaction['transaction_amount']:.2f}"
            st.session_state.fraud_alerts.insert(0, alert_msg)  # Add to the top
            if len(st.session_state.fraud_alerts) > 5:
                st.session_state.fraud_alerts.pop()  # Keep only the last 5

        with fraud_alert_placeholder.container():
            for alert in st.session_state.fraud_alerts:
                st.warning(alert)

        with placeholder.container():
            st.write("### 🔄 Latest Transactions")
            st.dataframe(st.session_state.transactions.tail(10))

        # 📊 Fraud distribution pie chart
        with fraud_chart_placeholder.container():
            fraud_counts = st.session_state.transactions['is_fraud'].value_counts()
            labels = fraud_counts.index.map(lambda x: "Legitimate" if x == 0 else "Fraud").tolist()
            colors = ['green' if label == "Legitimate" else 'red' for label in labels]
            fig, ax = plt.subplots()
            ax.pie(fraud_counts, labels=labels, autopct='%1.1f%%', colors=colors)
            st.pyplot(fig)

        # 📈 Transaction trend line chart
        with trends_chart_placeholder.container():
            fig, ax = plt.subplots()
            ax.plot(st.session_state.transactions.index, st.session_state.transactions['transaction_amount'], marker='o', label='Transaction Amount')
            ax.set_xlabel("Transaction Index")
            ax.set_ylabel("Amount (PKR)")
            ax.legend()
            st.pyplot(fig)

        # 📉 Fraud trend over time
        with fraud_trend_placeholder.container():
            fraud_counts = st.session_state.transactions.groupby(st.session_state.transactions.index)['is_fraud'].sum()
            fig, ax = plt.subplots()
            ax.bar(fraud_counts.index, fraud_counts.values, color='red', label="Fraudulent Transactions")
            ax.set_xlabel("Transaction Index")
            ax.set_ylabel("Fraud Count")
            ax.legend()
            st.pyplot(fig)

        # 🏪 Fraud vs. Legitimate per category
        with category_distribution_placeholder.container():
            category_counts = st.session_state.transactions.groupby(['merchant_category', 'is_fraud']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 5))
            category_counts.plot(kind='bar', stacked=True, ax=ax)
            st.pyplot(fig)

        # 💰 Average Transaction Amount (Fraud vs. Legitimate)
        with amount_comparison_placeholder.container():
            avg_amounts = st.session_state.transactions.groupby('is_fraud')['transaction_amount'].mean()
            fig, ax = plt.subplots()
            avg_amounts.plot(kind='bar', color=['green', 'red'], ax=ax)
            st.pyplot(fig)

    time.sleep(update_interval)
