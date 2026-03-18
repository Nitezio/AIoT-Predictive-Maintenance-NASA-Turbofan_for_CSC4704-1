import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_accuracy():
    # Load predictions
    preds = pd.read_csv('predictions.csv')
    
    # 1. Error Distribution
    preds['Error'] = preds['Predicted_RUL'] - preds['Actual_RUL']
    
    print("--- Statistical Accuracy Overview ---")
    print(preds['Error'].describe())
    
    # 2. Classification-like Metrics (Critical < 20 cycles)
    # This is important for "SME Resilience" - how well do we catch engines about to fail?
    threshold = 20
    preds['Actual_Critical'] = (preds['Actual_RUL'] < threshold).astype(int)
    preds['Predicted_Critical'] = (preds['Predicted_RUL'] < threshold).astype(int)
    
    print("\n--- Critical Failure Prediction (RUL < 20) ---")
    f1 = f1_score(preds['Actual_Critical'], preds['Predicted_Critical'])
    print(f"F1 Score (Critical Class): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(preds['Actual_Critical'], preds['Predicted_Critical'], target_names=['Healthy/Warning', 'Critical']))
    
    # 3. Error Distribution Visualization (Console Summary)
    bins = [-np.inf, -20, -10, 10, 20, np.inf]
    labels = ['Over-pessimistic (>20)', 'Slightly Pessimistic', 'Accurate (+/-10)', 'Slightly Optimistic', 'Dangerous Over-optimistic (>20)']
    error_cats = pd.cut(preds['Error'], bins=bins, labels=labels)
    print("\n--- Error Distribution Summary ---")
    print(error_cats.value_counts().sort_index())

if __name__ == "__main__":
    evaluate_accuracy()
