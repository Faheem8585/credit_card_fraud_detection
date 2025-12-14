from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, DECIMAL, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import os

# Database URL from environment variable or default to SQLite for development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./fraud_detection.db"  # SQLite for local development
)

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, echo=False)

# Base class
Base = declarative_base()

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="user")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    transaction_time = Column(DateTime, default=datetime.utcnow)
    amount = Column(DECIMAL(10, 2), nullable=False)
    
    # PCA features V1-V28
    v1 = Column(DECIMAL(10, 6))
    v2 = Column(DECIMAL(10, 6))
    v3 = Column(DECIMAL(10, 6))
    v4 = Column(DECIMAL(10, 6))
    v5 = Column(DECIMAL(10, 6))
    v6 = Column(DECIMAL(10, 6))
    v7 = Column(DECIMAL(10, 6))
    v8 = Column(DECIMAL(10, 6))
    v9 = Column(DECIMAL(10, 6))
    v10 = Column(DECIMAL(10, 6))
    v11 = Column(DECIMAL(10, 6))
    v12 = Column(DECIMAL(10, 6))
    v13 = Column(DECIMAL(10, 6))
    v14 = Column(DECIMAL(10, 6))
    v15 = Column(DECIMAL(10, 6))
    v16 = Column(DECIMAL(10, 6))
    v17 = Column(DECIMAL(10, 6))
    v18 = Column(DECIMAL(10, 6))
    v19 = Column(DECIMAL(10, 6))
    v20 = Column(DECIMAL(10, 6))
    v21 = Column(DECIMAL(10, 6))
    v22 = Column(DECIMAL(10, 6))
    v23 = Column(DECIMAL(10, 6))
    v24 = Column(DECIMAL(10, 6))
    v25 = Column(DECIMAL(10, 6))
    v26 = Column(DECIMAL(10, 6))
    v27 = Column(DECIMAL(10, 6))
    v28 = Column(DECIMAL(10, 6))
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="transactions")
    prediction = relationship("Prediction", back_populates="transaction", uselist=False)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    fraud_probability = Column(DECIMAL(5, 4), nullable=False)
    decision = Column(String(10), nullable=False)  # ALLOW, BLOCK, FLAG
    risk_level = Column(String(20), index=True)
    individual_predictions = Column(JSON)  # Store model predictions as JSON
    model_weights = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    transaction = relationship("Transaction", back_populates="prediction")
    fraud_alerts = relationship("FraudAlert", back_populates="prediction")

class FraudAlert(Base):
    __tablename__ = "fraud_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    severity = Column(String(20), index=True)  # low, medium, high, critical
    status = Column(String(20), default="pending", index=True)  # pending, reviewed, resolved, false_positive
    reviewed_by = Column(Integer, ForeignKey("users.id"))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="fraud_alerts")

# Create all tables
def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

# Get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
