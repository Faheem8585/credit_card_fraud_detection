import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_synthetic_data(n_samples=100000, fraud_ratio=0.002, random_state=42):
    """
    Generates a synthetic dataset mimicking the Kaggle Credit Card Fraud dataset.
    
    Args:
        n_samples (int): Total number of transactions.
        fraud_ratio (float): Percentage of fraudulent transactions.
        random_state (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame with features V1-V28, Time, Amount, Class.
    """
    np.random.seed(random_state)
    
    # Calculate number of fraud samples
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    print(f"Generating {n_samples} transactions...")
    print(f"Legitimate: {n_legit}, Fraudulent: {n_fraud} (Ratio: {fraud_ratio:.4f})")
    
    # Generate V1-V28 features using make_classification
    # We use 28 informative features to mimic the PCA components
    X, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=20,
        n_redundant=8,
        n_repeated=0,
        n_classes=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.001, # Add some noise
        random_state=random_state
    )
    
    # Create DataFrame
    cols = [f'V{i}' for i in range(1, 29)]
    df = pd.DataFrame(X, columns=cols)
    
    # Generate 'Time' feature (simulation of 2 days in seconds)
    # Fraud might happen more at weird times, but for now uniform random
    df['Time'] = np.random.uniform(0, 172800, n_samples)
    df['Time'] = df['Time'].sort_values().values # Sort by time
    
    # Generate 'Amount' feature
    # Fraud amounts often have different distribution (higher variance or specific amounts)
    # Legit amounts: Log-normal distribution
    amounts = np.random.lognormal(mean=2, sigma=1, size=n_samples)
    
    # Add some high-value fraud spikes
    fraud_indices = np.where(y == 1)[0]
    amounts[fraud_indices] *= np.random.uniform(1, 5, size=len(fraud_indices))
    
    df['Amount'] = amounts
    df['Class'] = y
    
    return df

if __name__ == "__main__":
    import os
    
    # Ensure data directory exists
    os.makedirs("credit_card_fraud_detection/data", exist_ok=True)
    
    # Generate data
    df = generate_synthetic_data(n_samples=200000, fraud_ratio=0.005)
    
    # Save to CSV
    output_path = "credit_card_fraud_detection/data/creditcard.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(df['Class'].value_counts())
