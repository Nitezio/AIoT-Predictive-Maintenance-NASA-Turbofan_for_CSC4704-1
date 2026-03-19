import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
import os

def evaluate_accuracy():
    # Load predictions
    if not os.path.exists('predictions.csv'):
        print("Error: predictions.csv not found. Run train_model.py first.")
        return
        
    preds = pd.read_csv('predictions.csv')
    
    # 1. Error Distribution
    preds['Error'] = preds['Predicted_RUL'] - preds['Actual_RUL']
    
    print("--- 📊 Statistical Accuracy Overview ---")
    print(preds['Error'].describe())
    
    # 2. MULTI-CLASS CATEGORIZATION (The Hackathon Standard)
    # Define Categories
    def categorize(rul):
        if rul < 20: return 0 # Critical
        if rul < 50: return 1 # Warning
        return 2 # Healthy

    preds['Actual_Cat'] = preds['Actual_RUL'].apply(categorize)
    preds['Predicted_Cat'] = preds['Predicted_RUL'].apply(categorize)
    
    print("\n--- 🛡️ Multi-Class Reliability (Healthy vs Warning vs Critical) ---")
    
    # Weighted F1 takes all classes into account
    f1_weighted = f1_score(preds['Actual_Cat'], preds['Predicted_Cat'], average='weighted')
    print(f"Overall System F1-Score (Weighted): {f1_weighted:.4f}")
    
    print("\nDetailed Classification Report:")
    target_names = ['🔴 CRITICAL', '🟡 WARNING', '🟢 HEALTHY']
    print(classification_report(preds['Actual_Cat'], preds['Predicted_Cat'], target_names=target_names))
    
    # 3. Safety Analysis
    dangerous = len(preds[(preds['Predicted_RUL'] - preds['Actual_RUL']) > 20])
    print(f"\n--- ⚠️ Safety Audit ---")
    print(f"Dangerous Over-optimistic Errors: {dangerous}")
    if dangerous < 10:
        print("✅ PASSED: Safety threshold met (<10 dangerous errors).")
    else:
        print("❌ FAILED: Safety threshold exceeded.")

if __name__ == "__main__":
    evaluate_accuracy()
