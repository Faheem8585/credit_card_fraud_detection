import pandas as pd
import joblib
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def train_ensemble_models(data_dir, model_dir, dashboard_assets_dir):
    """
    Trains multiple models and creates an ensemble.
    Models: Logistic Regression, Random Forest, Gradient Boosting, Isolation Forest
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

    models = {}
    results = {}

    # 1. Logistic Regression
    print("\n[1/4] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_val)[:, 1]
    auc_lr = roc_auc_score(y_val, y_prob_lr)
    print(f"✓ Logistic Regression AUC: {auc_lr:.4f}")
    models['lr'] = lr
    results['lr'] = auc_lr

    # 2. Random Forest
    print("\n[2/4] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_val)[:, 1]
    auc_rf = roc_auc_score(y_val, y_prob_rf)
    print(f"✓ Random Forest AUC: {auc_rf:.4f}")
    models['rf'] = rf
    results['rf'] = auc_rf

    # 3. Gradient Boosting (Sklearn's native implementation)
    print("\n[3/4] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_prob_gb = gb.predict_proba(X_val)[:, 1]
    auc_gb = roc_auc_score(y_val, y_prob_gb)
    print(f"✓ Gradient Boosting AUC: {auc_gb:.4f}")
    models['gb'] = gb
    results['gb'] = auc_gb

    # 4. Isolation Forest (Anomaly Detection)
    print("\n[4/4] Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    anomaly_scores = iso_forest.decision_function(X_val)
    # Normalize to [0, 1] probability range
    y_prob_iso = 1 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    auc_iso = roc_auc_score(y_val, y_prob_iso)
    print(f"✓ Isolation Forest AUC: {auc_iso:.4f}")
    models['iso_forest'] = iso_forest
    results['iso_forest'] = auc_iso

    # 5. Ensemble (Weighted Average)
    print("\n[ENSEMBLE] Creating Weighted Ensemble...")
    weights = {
        'lr': auc_lr,
        'rf': auc_rf,
        'gb': auc_gb,
        'iso_forest': auc_iso
    }
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Calculate weighted ensemble probability
    y_prob_ensemble = (
        weights['lr'] * y_prob_lr +
        weights['rf'] * y_prob_rf +
        weights['gb'] * y_prob_gb +
        weights['iso_forest'] * y_prob_iso
    )
    
    y_pred_ensemble = (y_prob_ensemble > 0.5).astype(int)
    auc_ensemble = roc_auc_score(y_val, y_prob_ensemble)
    print(f"✓ Ensemble AUC: {auc_ensemble:.4f}")
    print("\n📊 Model Weights:")
    for name, weight in weights.items():
        print(f"  • {name.upper()}: {weight:.4f}")

    # Detailed Report for Ensemble
    print("\n📈 Ensemble Performance:")
    print(classification_report(y_val, y_pred_ensemble))

    # Select Best Model
    all_results = {**results, 'ensemble': auc_ensemble}
    best_name = max(all_results, key=all_results.get)
    best_auc = all_results[best_name]
    
    print(f"\n🏆 Best Model: {best_name.upper()} (AUC: {best_auc:.4f})")
    
    # Save all models
    for name, model in models.items():
        model_path = os.path.join(model_dir, f'{name}_model.pkl')
        joblib.dump(model, model_path)
    print(f"\n💾 Saved {len(models)} models to {model_dir}")
    
    # Save ensemble weights and config
    ensemble_config = {
        'weights': weights,
        'best_model': best_name,
        'auc_scores': all_results,
        'model_list': list(models.keys())
    }
    with open(os.path.join(model_dir, 'ensemble_config.json'), 'w') as f:
        json.dump(ensemble_config, f, indent=4)
    
    # Save metrics
    metrics = {
        "ensemble_auc": float(auc_ensemble),
        "individual_aucs": {k: float(v) for k, v in results.items()},
        "report": classification_report(y_val, y_pred_ensemble, output_dict=True)
    }
    with open(os.path.join(model_dir, 'ensemble_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Saved ensemble config and metrics")

    # ========== COMPREHENSIVE VISUALIZATIONS ==========
    
    # 1. ROC Curves Comparison
    print("\n📊 Generating ROC curves...")
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    for idx, (name, prob) in enumerate([
        ('lr', y_prob_lr),
        ('rf', y_prob_rf),
        ('gb', y_prob_gb),
        ('iso_forest', y_prob_iso)
    ]):
        fpr, tpr, _ = roc_curve(y_val, prob)
        plt.plot(fpr, tpr, label=f'{name.upper()} (AUC={results[name]:.4f})', 
                color=colors[idx], linewidth=2)
    
    # Ensemble ROC
    fpr, tpr, _ = roc_curve(y_val, y_prob_ensemble)
    plt.plot(fpr, tpr, label=f'ENSEMBLE (AUC={auc_ensemble:.4f})', 
            linewidth=3, linestyle='--', color='red')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves: Multi-Model Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dashboard_assets_dir, 'roc_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Model AUC Comparison Bar Chart
    print("📊 Generating model comparison...")
    plt.figure(figsize=(12, 7))
    names = list(all_results.keys())
    aucs = list(all_results.values())
    colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = plt.barh(names, aucs, color=colors_bar[:len(names)])
    
    best_idx = names.index(best_name)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    
    plt.xlabel('AUC Score', fontsize=14)
    plt.title('Model Performance Comparison (AUC)', fontsize=16, fontweight='bold')
    plt.xlim(0, 1)
    for i, (name, auc) in enumerate(zip(names, aucs)):
        label = f'{auc:.4f}'
        if name == best_name:
            label += ' 🏆'
        plt.text(auc + 0.01, i, label, va='center', fontsize=11,
                fontweight='bold' if name == best_name else 'normal')
    plt.tight_layout()
    plt.savefig(os.path.join(dashboard_assets_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrices
    print("📊 Generating confusion matrices...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrices for All Models', fontsize=18, fontweight='bold')
    
    model_preds = {
        'LR': (y_prob_lr > 0.5).astype(int),
        'RF': (y_prob_rf > 0.5).astype(int),
        'GB': (y_prob_gb > 0.5).astype(int),
        'Iso': (y_prob_iso > 0.5).astype(int),
        'Ensemble': y_pred_ensemble
    }
    
    for idx, (name, preds) in enumerate(model_preds.items()):
        ax = axes[idx // 3, idx % 3]
        cm = confusion_matrix(y_val, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dashboard_assets_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Curves
    print("📊 Generating precision-recall curves...")
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    plt.figure(figsize=(12, 8))
    
    for idx, (name, prob) in enumerate([
        ('lr', y_prob_lr),
        ('rf', y_prob_rf),
        ('gb', y_prob_gb),
        ('iso_forest', y_prob_iso)
    ]):
        precision, recall, _ = precision_recall_curve(y_val, prob)
        ap = average_precision_score(y_val, prob)
        plt.plot(recall, precision, label=f'{name.upper()} (AP={ap:.4f})', 
                color=colors[idx], linewidth=2)
    
    # Ensemble
    precision, recall, _ = precision_recall_curve(y_val, y_prob_ensemble)
    ap = average_precision_score(y_val, y_prob_ensemble)
    plt.plot(recall, precision, label=f'ENSEMBLE (AP={ap:.4f})', 
            linewidth=3, linestyle='--', color='red')
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(dashboard_assets_dir, 'precision_recall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Detailed Metrics Table
    print("📊 Generating metrics table...")
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    metrics_data = []
    for name, prob in [('LR', y_prob_lr), ('RF', y_prob_rf), ('GB', y_prob_gb), 
                       ('Iso', y_prob_iso), ('Ensemble', y_prob_ensemble)]:
        preds = (prob > 0.5).astype(int)
        metrics_data.append({
            'Model': name,
            'AUC': roc_auc_score(y_val, prob),
            'Accuracy': accuracy_score(y_val, preds),
            'Precision': precision_score(y_val, preds),
            'Recall': recall_score(y_val, preds),
            'F1-Score': f1_score(y_val, preds)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Format cell values
    cell_text = []
    for row in metrics_df.values:
        formatted_row = [row[0]]  # Model name (string)
        formatted_row.extend([f'{val:.4f}' for val in row[1:]])  # Numeric values
        cell_text.append(formatted_row)
    
    table = ax.table(cellText=cell_text, 
                    colLabels=metrics_df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(metrics_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    for i, row in enumerate(metrics_df.values):
        for j in range(1, len(row)):
            table[(i+1, j)].set_facecolor('#ecf0f1')
    
    plt.title('Detailed Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(dashboard_assets_dir, 'metrics_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✅ All visualizations generated!")
    print(f"   - roc_comparison.png")
    print(f"   - model_comparison.png")
    print(f"   - confusion_matrices.png")
    print(f"   - precision_recall.png")
    print(f"   - metrics_table.png")
    
    print("\n✅ Ensemble training complete!")

if __name__ == "__main__":
    train_ensemble_models(
        "credit_card_fraud_detection/data/processed",
        "credit_card_fraud_detection/models",
        "credit_card_fraud_detection/dashboard/assets"
    )
