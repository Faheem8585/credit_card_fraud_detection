-- Credit Card Fraud Detection Database Schema
-- Production-ready PostgreSQL schema for full-stack application

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create index on email for fast lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    transaction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    amount DECIMAL(10, 2) NOT NULL,
    -- Feature columns (V1-V28 from PCA)
    v1 DECIMAL(10, 6),
    v2 DECIMAL(10, 6),
    v3 DECIMAL(10, 6),
    v4 DECIMAL(10, 6),
    v5 DECIMAL(10, 6),
    v6 DECIMAL(10, 6),
    v7 DECIMAL(10, 6),
    v8 DECIMAL(10, 6),
    v9 DECIMAL(10, 6),
    v10 DECIMAL(10, 6),
    v11 DECIMAL(10, 6),
    v12 DECIMAL(10, 6),
    v13 DECIMAL(10, 6),
    v14 DECIMAL(10, 6),
    v15 DECIMAL(10, 6),
    v16 DECIMAL(10, 6),
    v17 DECIMAL(10, 6),
    v18 DECIMAL(10, 6),
    v19 DECIMAL(10, 6),
    v20 DECIMAL(10, 6),
    v21 DECIMAL(10, 6),
    v22 DECIMAL(10, 6),
    v23 DECIMAL(10, 6),
    v24 DECIMAL(10, 6),
    v25 DECIMAL(10, 6),
    v26 DECIMAL(10, 6),
    v27 DECIMAL(10, 6),
    v28 DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions(transaction_time);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    transaction_id INTEGER REFERENCES transactions(id) ON DELETE CASCADE,
    fraud_probability DECIMAL(5, 4) NOT NULL,
    decision VARCHAR(10) CHECK (decision IN ('ALLOW', 'BLOCK', 'FLAG')),
    risk_level VARCHAR(20),
    -- Store individual model predictions as JSON
    individual_predictions JSONB,
    model_weights JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_decision ON predictions(decision);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level);

-- Fraud alerts table (for admin monitoring)
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id SERIAL PRIMARY KEY,
    transaction_id INTEGER REFERENCES transactions(id),
    prediction_id INTEGER REFERENCES predictions(id),
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed', 'resolved', 'false_positive')),
    reviewed_by INTEGER REFERENCES users(id),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status ON fraud_alerts(status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_severity ON fraud_alerts(severity);

-- Create view for admin dashboard stats
CREATE OR REPLACE VIEW admin_dashboard_stats AS
SELECT 
    (SELECT COUNT(*) FROM transactions) AS total_transactions,
    (SELECT COUNT(*) FROM predictions WHERE decision = 'BLOCK') AS blocked_transactions,
    (SELECT AVG(fraud_probability) FROM predictions) AS avg_fraud_probability,
    (SELECT COUNT(*) FROM fraud_alerts WHERE status = 'pending') AS pending_alerts,
    (SELECT COUNT(*) FROM users) AS total_users;

-- NOTE: Create admin user manually via signup endpoint with role='admin'
-- Or run: INSERT INTO users (email, password_hash, role) VALUES ('admin@fraud.com', '<bcrypt_hash>', 'admin');
