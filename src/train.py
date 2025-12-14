import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def train_models(data_dir, model_dir, dashboard_assets_dir):
    """
    Trains Logistic Regression and Random Forest models.
    Evaluates and saves the best one.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(dashboard_assets_dir):
        os.makedirs(dashboard_assets_dir)

    print("Loading processed data...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv')).values.ravel()

    # 1. Logistic Regression (Baseline)
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    y_pred_lr = lr.predict(X_val)
    y_prob_lr = lr.predict_proba(X_val)[:, 1]
    auc_lr = roc_auc_score(y_val, y_prob_lr)
    print(f"Logistic Regression AUC: {auc_lr:.4f}")
    print(classification_report(y_val, y_pred_lr))

    # 2. Random Forest (Champion)
    print("\nTraining Random Forest (this may take a moment)...")
    # Reduced n_estimators for speed in demo, increase for prod
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_val)
    y_prob_rf = rf.predict_proba(X_val)[:, 1]
    auc_rf = roc_auc_score(y_val, y_prob_rf)
    print(f"Random Forest AUC: {auc_rf:.4f}")
    print(classification_report(y_val, y_pred_rf))

    # Select Best Model
    if auc_rf > auc_lr:
        best_model = rf
        best_name = "Random Forest"
        y_prob_best = y_prob_rf
        y_pred_best = y_pred_rf
    else:
        best_model = lr
        best_name = "Logistic Regression"
        y_prob_best = y_prob_lr
        y_pred_best = y_pred_lr

    print(f"\nBest Model: {best_name}")
    
    # Save Model
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Saved model to {model_path}")

    # Save Metrics
    metrics = {
        "model": best_name,
        "auc": float(max(auc_rf, auc_lr)),
        "report": classification_report(y_val, y_pred_best, output_dict=True)
    }
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Generate Plots for Dashboard
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_val, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({best_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(dashboard_assets_dir, 'confusion_matrix_model.png'))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_prob_best)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{best_name} (AUC = {max(auc_rf, auc_lr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(dashboard_assets_dir, 'roc_curve.png'))
    print("Saved evaluation plots.")

if __name__ == "__main__":
    train_models(
        "credit_card_fraud_detection/data/processed",
        "credit_card_fraud_detection/models",
        "credit_card_fraud_detection/dashboard/assets"
    )
