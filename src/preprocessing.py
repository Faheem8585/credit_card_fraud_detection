import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_data(input_path, output_dir):
    """
    Loads data, scales features, splits into train/val/test, and applies SMOTE.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud count: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    # 1. Split Data (Train/Val/Test)
    # Stratified split to maintain fraud ratio
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # First split: 80% Train+Val, 20% Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: 80% Train, 20% Val (from the 80% temp)
    # Effective: 64% Train, 16% Val, 20% Test
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 2. Scaling
    # RobustScaler is less prone to outliers (good for 'Amount')
    scaler = RobustScaler()
    
    # Fit ONLY on training data
    cols_to_scale = ['Time', 'Amount']
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # Save scaler for inference
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # 3. Handle Imbalance (SMOTE)
    # Apply ONLY to Training data
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled Train Shape: {X_train_resampled.shape}")
    print(f"Resampled Class Dist: {y_train_resampled.value_counts().to_dict()}")
    
    # 4. Save Processed Data
    print("Saving processed datasets...")
    X_train_resampled.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    y_train_resampled.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    
    X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_data(
        "credit_card_fraud_detection/data/creditcard.csv",
        "credit_card_fraud_detection/data/processed"
    )
