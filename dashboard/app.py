import streamlit as st
import requests
import pandas as pd
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
ASSETS_DIR = "assets"

st.set_page_config(page_title="Fraud Detection - Full Stack", layout="wide", page_icon="🛡️")

# Initialize session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None

# Sidebar - Authentication
with st.sidebar:
    st.title("🛡️ Fraud Detection")
    st.markdown("---")
    
    if not st.session_state.token:
        st.subheader("Login / Signup")
        
        tab1, tab2 = st.tabs(["Login", "Signup"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("Login")
                
                if login_btn:
                    try:
                        response = requests.post(
                            f"{API_URL}/auth/login",
                            json={"email": email, "password": password}
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.token = data['access_token']
                            st.session_state.user = {
                                'email': data['email'],
                                'user_id': data['user_id'],
                                'role': data['role']
                            }
                            st.success("Logged in!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                signup_btn = st.form_submit_button("Create Account")
                
                if signup_btn:
                    try:
                        response = requests.post(
                            f"{API_URL}/auth/signup",
                            json={"email": new_email, "password": new_password}
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.token = data['access_token']
                            st.session_state.user = {
                                'email': data['email'],
                                'user_id': data['user_id'],
                                'role': data['role']
                            }
                            st.success("Account created!")
                            st.rerun()
                        else:
                            st.error(f"Error: {response.json().get('detail', 'Signup failed')}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.success(f"👤 {st.session_state.user['email']}")
        st.caption(f"Role: {st.session_state.user['role']}")
        
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.user = None
            st.rerun()

# Main Content
if not st.session_state.token:
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("### Full-Stack Application with Real Kaggle Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model AUC", "0.9766", "+3% vs synthetic")
    with col2:
        st.metric("Dataset Size", "284,807", "transactions")
    with col3:
        st.metric("Fraud Rate", "0.17%", "real-world")
    
    st.markdown("---")
    st.info("👈 Please **Login** or **Signup** in the sidebar to access the fraud detection system")
    
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **Multi-Model Ensemble** (Gradient Boosting, Random Forest, LR, Isolation Forest)
    - **JWT Authentication** (secure access)
    - **Database Logging** (all predictions saved)
    - **Transaction History** (search your past transactions)
    - **Admin Dashboard** (monitor system-wide stats)
    - **Real-Time Detection** (< 100ms latency)
    """)

else:
    # Navigation
    page = st.sidebar.radio("Navigation", [
        "🔍 Fraud Detection", 
        "📊 Model Performance",
        "🔴 Live Monitor",
        "📋 My Transactions", 
        "⚙️ Admin Panel"
    ])
    
    if page == "🔍 Fraud Detection":
        st.title("🔍 Fraud Detection")
        st.markdown("Submit a transaction for fraud analysis")
        
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            with col1:
                amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
                time_val = st.number_input("Time (Seconds)", min_value=0.0, value=0.0)
            
            with col2:
                st.markdown("**PCA Features (V1-V28)**")
                v_preset = st.selectbox("Use Preset", ["Normal", "Suspicious", "Custom"])
                if v_preset == "Suspicious":
                    v_mean = -3.0
                elif v_preset == "Normal":
                    v_mean = 0.0
                else:
                    v_mean = st.slider("V1-V28 Mean", -5.0, 5.0, 0.0)
            
            submit = st.form_submit_button("🔎 Analyze Transaction", use_container_width=True)
        
        if submit:
            # Construct payload
            payload = {"Time": time_val, "Amount": amount}
            for i in range(1, 29):
                payload[f"V{i}"] = v_mean
            
            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                response = requests.post(f"{API_URL}/predict", json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    
                    # Main Result
                    st.markdown("### 🎯 Analysis Result")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Ensemble Score", f"{result['fraud_probability']:.4f}")
                    col2.metric("Risk Level", result['risk_level'])
                    col3.metric("Decision", result['decision'])
                    col4.metric("Transaction ID", f"#{result['transaction_id']}")
                    
                    # Individual Models
                    st.markdown("---")
                    st.markdown("### 🤖 Model Predictions")
                    
                    model_cols = st.columns(4)
                    individual = result['individual_predictions']
                    weights = result['model_weights']
                    
                    for idx, (model_name, prob) in enumerate(individual.items()):
                        with model_cols[idx]:
                            weight = weights.get(model_name, 0)
                            st.metric(
                                label=model_name.upper(),
                                value=f"{prob:.4f}",
                                delta=f"Weight: {weight:.1%}",
                                delta_color="off"
                            )
                            if prob > 0.5:
                                st.error("🚨 Fraud")
                            else:
                                st.success("✅ Legit")
                    
                    # Verdict
                    st.markdown("---")
                    if result['is_fraud']:
                        st.error("🚨 **HIGH RISK**: Transaction flagged as fraudulent and BLOCKED")
                    else:
                        st.success("✅ **APPROVED**: Transaction appears legitimate")
                        
                elif response.status_code == 401:
                    st.error("Session expired. Please login again.")
                    st.session_state.token = None
                    st.rerun()
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif page == "📊 Model Performance":
        st.title("📊 Model Performance")
        st.markdown("Performance metrics on real Kaggle data (284,807 transactions)")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", "Gradient Boosting")
        col2.metric("Ensemble AUC", "0.9766", "+3% vs synthetic")
        col3.metric("Fraud Rate", "0.17%", "real-world")
        col4.metric("Models", "4", "LR, RF, GB, Iso")
        
        st.markdown("---")
        
        # Try to show plots if they exist
        import os
        # Use absolute path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, "assets")
        
        # ROC Comparison
        roc_path = os.path.join(assets_dir, "roc_comparison.png")
        if os.path.exists(roc_path):
            st.subheader("🎯 ROC Curves Comparison")
            st.image(roc_path, use_container_width=True)
            st.caption("Receiver Operating Characteristic curves showing True Positive Rate vs False Positive Rate")
        
        # Model AUC Comparison  
        comp_path = os.path.join(assets_dir, "model_comparison.png")
        if os.path.exists(comp_path):
            st.subheader("📊 Model AUC Comparison")
            st.image(comp_path, use_container_width=True)
            st.caption("Area Under the ROC Curve (AUC) for each model")
        
        # Confusion Matrices
        cm_path = os.path.join(assets_dir, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.subheader("🔢 Confusion Matrices")
            st.image(cm_path, use_container_width=True)
            st.caption("True Positives, False Positives, True Negatives, False Negatives for each model")
        
        # Precision-Recall Curves
        pr_path = os.path.join(assets_dir, "precision_recall.png")
        if os.path.exists(pr_path):
            st.subheader("⚖️ Precision-Recall Curves")
            st.image(pr_path, use_container_width=True)
            st.caption("Trade-off between Precision and Recall at different thresholds")
        
        # Metrics Table
        metrics_path = os.path.join(assets_dir, "metrics_table.png")
        if os.path.exists(metrics_path):
            st.subheader("📋 Detailed Metrics Comparison")
            st.image(metrics_path, use_container_width=True)
            st.caption("Comprehensive comparison of all evaluation metrics")
        
        # If no plots exist
        if not any(os.path.exists(p) for p in [roc_path, comp_path, cm_path, pr_path, metrics_path]):
            st.info("📈 Visualizations are being generated. Please wait for model training to complete...")
            st.markdown("The system is currently training on 284,807 real transactions from Kaggle.")
    
    elif page == "🔴 Live Monitor":
        st.title("🔴 Live Transaction Monitor")
        st.markdown("Simulated real-time fraud detection")
        
        # Simulation controls
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("▶️ Start Simulation", use_container_width=True):
                st.session_state.monitoring = True
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.monitoring = False
        
        if 'monitoring' not in st.session_state:
            st.session_state.monitoring = False
        if 'monitor_data' not in st.session_state:
            st.session_state.monitor_data = []
        
        if st.session_state.monitoring:
            import time
            import random
            
            # Simulate transaction
            amount = random.uniform(1, 2000)
            is_suspicious = random.random() < 0.1
            v_mean = -3.0 if is_suspicious else random.uniform(-1, 1)
            
            payload = {"Time": random.randint(0, 172800), "Amount": amount}
            for i in range(1, 29):
                payload[f"V{i}"] = v_mean + random.uniform(-0.5, 0.5)
            
            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                response = requests.post(f"{API_URL}/predict", json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Add to monitor data
                    st.session_state.monitor_data.insert(0, {
                        'time': pd.Timestamp.now().strftime("%H:%M:%S"),
                        'amount': f"${amount:.2f}",
                        'fraud_prob': f"{result['fraud_probability']:.3f}",
                        'decision': result['decision'],
                        'risk': result['risk_level']
                    })
                    
                    # Keep last 20
                    st.session_state.monitor_data = st.session_state.monitor_data[:20]
                    
                    # Display
                    st.subheader("Live Transactions")
                    df = pd.DataFrame(st.session_state.monitor_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Simulation error: {e}")
                st.session_state.monitoring = False
        else:
            if st.session_state.monitor_data:
                st.subheader("Transaction History")
                df = pd.DataFrame(st.session_state.monitor_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Click 'Start Simulation' to begin monitoring")
    
    elif page == "📋 My Transactions":
        st.title("📊 Transaction History")
        
        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            response = requests.get(f"{API_URL}/transactions?limit=50", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                count = data.get('count', 0)
                
                st.metric("Total Transactions", count)
                
                if count > 0:
                    st.info("📝 Your transaction history is stored in the database")
                    st.json(data)
                else:
                    st.info("No transactions yet. Submit your first transaction in the Fraud Detection page!")
            else:
                st.error("Failed to load transactions")
        except Exception as e:
            st.error(f"Error: {e}")
    
    elif page == "⚙️ Admin Panel":
        if st.session_state.user['role'] != 'admin':
            st.warning("🔒 Admin access required")
            st.info("Default admin credentials:\n- Email: admin@fraud-detection.com\n- Password: admin123")
        else:
            st.title("⚙️ Admin Dashboard")
            
            try:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                response = requests.get(f"{API_URL}/admin/stats", headers=headers)
                
                if response.status_code == 200:
                    stats = response.json()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Transactions", stats['total_transactions'])
                    col2.metric("Total Users", stats['total_users'])
                    col3.metric("Blocked Txns", stats['blocked_transactions'], delta_color="inverse")
                    col4.metric("Fraud Rate", f"{stats['fraud_rate']:.2f}%")
                    
                    st.markdown("---")
                    
                    # Alerts
                    st.subheader("🚨 Pending Fraud Alerts")
                    alerts_response = requests.get(f"{API_URL}/admin/alerts", headers=headers)
                    if alerts_response.status_code == 200:
                        alerts_data = alerts_response.json()
                        st.metric("Pending Alerts", stats['pending_alerts'])
                        if stats['pending_alerts'] > 0:
                            st.json(alerts_data)
                    
                else:
                    st.error("Failed to load admin stats")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("🛡️ Full-Stack Fraud Detection System | Real Kaggle Data (AUC: 0.9766) | Powered by Multi-Model Ensemble")
