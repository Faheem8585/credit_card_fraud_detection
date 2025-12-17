from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_db, init_db, User
from database import crud
from auth.auth import create_access_token, verify_token
import joblib
import pandas as pd
import numpy as np
import json
from sqlalchemy.orm import Session
from datetime import timedelta
import uvicorn

# Initialize App
app = FastAPI(
    title="Credit Card Fraud Detection API - Full Stack",
    description="Production-ready fraud detection with authentication and database logging",
    version="3.0.0"
)

# Security
security = HTTPBearer()

# Load Models and Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")

models = {}
ensemble_config = {}
scaler = None

try:
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Load ensemble config
    with open(os.path.join(MODEL_DIR, 'ensemble_config.json'), 'r') as f:
        ensemble_config = json.load(f)
    
    # Load all models
    for model_name in ensemble_config['model_list']:
        model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
        models[model_name] = joblib.load(model_path)
    
    print(f"✅ Loaded {len(models)} models successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Pydantic Models
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    role: str

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

class PredictionResponse(BaseModel):
    transaction_id: int
    prediction_id: int
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    decision: str
    individual_predictions: dict
    model_weights: dict

# Dependency: Get current user from token
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Verify JWT token and return current user"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    user = crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    return user

# Routes
@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API - Full Stack",
        "version": "3.0.0",
        "features": [
            "JWT Authentication",
            "Database Logging",
            "Multi-Model Ensemble (AUC: 0.9766)",
            "Real Kaggle Data (284,807 transactions)"
        ],
        "ensemble_auc": ensemble_config.get('auc_scores', {}).get('ensemble'),
        "best_model": ensemble_config.get('best_model')
    }

# Authentication Endpoints
@app.post("/auth/signup", response_model=TokenResponse)
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """Create a new user account"""
    # Check if user exists
    existing_user = crud.get_user_by_email(db, request.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = crud.create_user(db, request.email, request.password)
    
    # Generate token
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id, "role": user.role}
    )
    
    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        role=user.role
    )

@app.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Login with email and password"""
    user = crud.authenticate_user(db, request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Generate token
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id, "role": user.role}
    )
    
    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        role=user.role
    )

@app.get("/auth/me")
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }

# Prediction Endpoint (Protected)
@app.post("/predict", response_model=PredictionResponse)
def predict(
    transaction: Transaction,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Make a fraud prediction and log to database.
    Requires authentication.
    """
    if not models or not scaler:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Convert input to DataFrame
    data = transaction.dict()
    df = pd.DataFrame([data])
    
    # Scale Time and Amount
    cols_to_scale = ['Time', 'Amount']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    # Reorder columns to match training data
    feature_order = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    df = df[feature_order]
    
    # Convert to numpy array to avoid feature name issues
    X = df.values
    
    # Get predictions from all models
    predictions = {}
    weights = ensemble_config['weights']
    
    for name, model in models.items():
        if name == 'iso_forest':
            anomaly_score = model.decision_function(X)[0]
            prob = 1 / (1 + np.exp(anomaly_score))
        else:
            prob = model.predict_proba(X)[0][1]
        
        predictions[name] = float(prob)
    
    # Calculate weighted ensemble probability
    ensemble_prob = sum(predictions[name] * weights[name] for name in predictions.keys())
    
    # Final prediction
    is_fraud = int(ensemble_prob > 0.5)
    
    # Rule-based Layer
    risk_level = "Low"
    if ensemble_prob > 0.8:
        risk_level = "Critical"
    elif ensemble_prob > 0.6:
        risk_level = "High"
    elif ensemble_prob > 0.4:
        risk_level = "Medium"
    elif data['Amount'] > 5000:
        risk_level = "Medium (High Amount)"
    
    decision = "BLOCK" if is_fraud else "ALLOW"
    
    # Log to database
    # 1. Create transaction record
    db_transaction = crud.create_transaction(db, current_user.id, data)
    
    # 2. Create prediction record
    prediction_data = {
        "fraud_probability": float(ensemble_prob),
        "decision": decision,
        "risk_level": risk_level,
        "individual_predictions": predictions,
        "model_weights": weights
    }
    db_prediction = crud.create_prediction(db, db_transaction.id, prediction_data)
    
    return PredictionResponse(
        transaction_id=db_transaction.id,
        prediction_id=db_prediction.id,
        fraud_probability=float(ensemble_prob),
        is_fraud=bool(is_fraud),
        risk_level=risk_level,
        decision=decision,
        individual_predictions=predictions,
        model_weights=weights
    )

# Transaction History
@app.get("/transactions")
def get_transactions(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's transaction history"""
    transactions = crud.get_user_transactions(db, current_user.id, limit, offset)
    return {
        "count": len(transactions),
        "transactions": transactions
    }

# Admin Routes
@app.get("/admin/stats")
def get_admin_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get admin dashboard statistics (Admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    stats = crud.get_admin_stats(db)
    return stats

@app.get("/admin/alerts")
def get_fraud_alerts(
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get pending fraud alerts (Admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    alerts = crud.get_pending_fraud_alerts(db, limit)
    return {"count": len(alerts), "alerts": alerts}

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    print("Initializing database...")
    try:
        init_db()
        print("✅ Database initialized")
        
        # Create admin users automatically
        print("Setting up admin users...")
        create_admin_users()
        
    except Exception as e:
        print(f"⚠️  Database init: {e}")

def create_admin_users():
    """Create default admin users on startup"""
    from sqlalchemy.orm import Session
    db = next(get_db())
    
    admin_accounts = [
        {"email": "admin@fraud-detection.com", "password": "admin123"},
        {"email": "admin2@fraud.com", "password": "admin123"}
    ]
    
    try:
        for admin_info in admin_accounts:
            existing_admin = crud.get_user_by_email(db, admin_info["email"])
            
            if existing_admin:
                # Update role to admin if not already
                if existing_admin.role != "admin":
                    existing_admin.role = "admin"
                    db.commit()
                    print(f"✅ Updated {admin_info['email']} to admin role")
                else:
                    print(f"✅ Admin user exists: {admin_info['email']}")
            else:
                # Create new admin user
                crud.create_user(db, admin_info["email"], admin_info["password"], role="admin")
                print(f"✅ Created admin user: {admin_info['email']}")
        
        print("🎉 Admin setup complete!")
        
    except Exception as e:
        print(f"⚠️  Admin setup error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
