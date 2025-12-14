from sqlalchemy.orm import Session
from database.db import User, Transaction, Prediction, FraudAlert
from auth.auth import hash_password, verify_password
from typing import Optional, List
from datetime import datetime

# User CRUD operations
def create_user(db: Session, email: str, password: str, role: str = "user") -> User:
    """Create a new user"""
    hashed_password = hash_password(password)
    user = User(
        email=email,
        password_hash=hashed_password,
        role=role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password"""
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    return user

# Transaction CRUD operations
def create_transaction(db: Session, user_id: int, transaction_data: dict) -> Transaction:
    """Create a new transaction"""
    transaction = Transaction(
        user_id=user_id,
        amount=transaction_data.get('Amount'),
        v1=transaction_data.get('V1'),
        v2=transaction_data.get('V2'),
        v3=transaction_data.get('V3'),
        v4=transaction_data.get('V4'),
        v5=transaction_data.get('V5'),
        v6=transaction_data.get('V6'),
        v7=transaction_data.get('V7'),
        v8=transaction_data.get('V8'),
        v9=transaction_data.get('V9'),
        v10=transaction_data.get('V10'),
        v11=transaction_data.get('V11'),
        v12=transaction_data.get('V12'),
        v13=transaction_data.get('V13'),
        v14=transaction_data.get('V14'),
        v15=transaction_data.get('V15'),
        v16=transaction_data.get('V16'),
        v17=transaction_data.get('V17'),
        v18=transaction_data.get('V18'),
        v19=transaction_data.get('V19'),
        v20=transaction_data.get('V20'),
        v21=transaction_data.get('V21'),
        v22=transaction_data.get('V22'),
        v23=transaction_data.get('V23'),
        v24=transaction_data.get('V24'),
        v25=transaction_data.get('V25'),
        v26=transaction_data.get('V26'),
        v27=transaction_data.get('V27'),
        v28=transaction_data.get('V28')
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return transaction

def get_user_transactions(db: Session, user_id: int, limit: int = 50, offset: int = 0) -> List[Transaction]:
    """Get transactions for a specific user"""
    return db.query(Transaction).filter(Transaction.user_id == user_id).order_by(Transaction.created_at.desc()).offset(offset).limit(limit).all()

# Prediction CRUD operations
def create_prediction(db: Session, transaction_id: int, prediction_data: dict) -> Prediction:
    """Create a new prediction result"""
    prediction = Prediction(
        transaction_id=transaction_id,
        fraud_probability=prediction_data.get('fraud_probability'),
        decision=prediction_data.get('decision'),
        risk_level=prediction_data.get('risk_level'),
        individual_predictions=prediction_data.get('individual_predictions'),
        model_weights=prediction_data.get('model_weights')
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    
    # Create fraud alert if high risk
    if prediction.fraud_probability > 0.8:
        create_fraud_alert(db, transaction_id, prediction.id, 'critical')
    elif prediction.fraud_probability > 0.6:
        create_fraud_alert(db, transaction_id, prediction.id, 'high')
    
    return prediction

# Fraud Alert CRUD operations
def create_fraud_alert(db: Session, transaction_id: int, prediction_id: int, severity: str) -> FraudAlert:
    """Create a fraud alert"""
    alert = FraudAlert(
        transaction_id=transaction_id,
        prediction_id=prediction_id,
        severity=severity
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert

def get_pending_fraud_alerts(db: Session, limit: int = 100) -> List[FraudAlert]:
    """Get pending fraud alerts"""
    return db.query(FraudAlert).filter(FraudAlert.status == 'pending').order_by(FraudAlert.created_at.desc()).limit(limit).all()

# Admin stats
def get_admin_stats(db: Session) -> dict:
    """Get statistics for admin dashboard"""
    total_transactions = db.query(Transaction).count()
    total_users = db.query(User).count()
    blocked_transactions = db.query(Prediction).filter(Prediction.decision == 'BLOCK').count()
    pending_alerts = db.query(FraudAlert).filter(FraudAlert.status == 'pending').count()
    
    return {
        "total_transactions": total_transactions,
        "total_users": total_users,
        "blocked_transactions": blocked_transactions,
        "pending_alerts": pending_alerts,
        "fraud_rate": (blocked_transactions / total_transactions * 100) if total_transactions > 0 else 0
    }
