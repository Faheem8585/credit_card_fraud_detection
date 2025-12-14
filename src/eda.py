import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(file_path, output_dir):
    """
    Reads the dataset and generates EDA plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution (0: Legit, 1: Fraud)')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    print("Saved class_distribution.png")
    
    # 2. Amount Distribution (Log scale)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Amount'], bins=50, kde=True, log_scale=True)
    plt.title('Transaction Amount Distribution (Log Scale)')
    plt.savefig(os.path.join(output_dir, 'amount_distribution.png'))
    print("Saved amount_distribution.png")
    
    # 3. Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20})
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    print("Saved correlation_matrix.png")
    
    # 4. Time vs Amount vs Class
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Time', y='Amount', hue='Class', data=df, alpha=0.5)
    plt.title('Time vs Amount (Colored by Class)')
    plt.savefig(os.path.join(output_dir, 'time_amount_class.png'))
    print("Saved time_amount_class.png")

if __name__ == "__main__":
    perform_eda(
        "credit_card_fraud_detection/data/creditcard.csv",
        "credit_card_fraud_detection/dashboard/assets"
    )
