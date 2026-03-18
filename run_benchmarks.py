import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
import joblib
import os
import subprocess
import json

# We will modify the seed in train_model.py dynamically or just re-implement the core loop here for speed and precision
from train_model import load_data, feature_engineering, prepare_train_data, prepare_test_data
from sklearn.ensemble import HistGradientBoostingRegressor

def run_trial(seed):
    # 1. Load and Process
    train_df, test_df, truth_df = load_data()
    train_processed = feature_engineering(train_df)
    test_processed = feature_engineering(test_df)
    train_final = prepare_train_data(train_processed)
    test_final = prepare_test_data(test_processed, truth_df)

    SELECTED_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
                        'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
                        'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
    features = SELECTED_SENSORS + ['Machine_State']
    for s in SELECTED_SENSORS:
        features.append(f'{s}_ema')
        for w in [10, 30]:
            features.append(f'{s}_mean_{w}')
            features.append(f'{s}_std_{w}')
            features.append(f'{s}_range_{w}')

    # 2. Train with specific seed
    weights = 1.0 / (train_final['RUL'] + 10)
    model = HistGradientBoostingRegressor(
        max_iter=500, max_depth=12, learning_rate=0.03,
        l2_regularization=0.5, random_state=seed
    )
    model.fit(train_final[features], train_final['RUL'], sample_weight=weights)

    # 3. Predict & Evaluate
    predictions = np.clip(model.predict(test_final[features]), 0, 125)
    actuals = test_final['RUL']
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    bias = (predictions - actuals).mean()
    
    # Critical F1
    threshold = 20
    actual_crit = (actuals < threshold).astype(int)
    pred_crit = (predictions < threshold).astype(int)
    f1 = f1_score(actual_crit, pred_crit)
    
    # Dangerous Over-optimism
    dangerous = len(predictions[(predictions - actuals) > 20])

    return {
        "Seed": seed,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
        "Bias": round(bias, 2),
        "F1 (Critical)": round(f1, 4),
        "Dangerous Errors": dangerous
    }

def main():
    seeds = [42, 123, 456, 789, 101112]
    results = []
    print(f"Starting 5-Trial Benchmark for High-Dependability Engine...")
    
    for i, s in enumerate(seeds):
        print(f"   Running Trial {i+1}/5 (Seed: {s})...")
        res = run_trial(s)
        results.append(res)
    
    df = pd.DataFrame(results)
    
    # Calculate Averages
    avg = df.mean(numeric_only=True).to_dict()
    avg["Seed"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    
    print("\n--- FINAL BENCHMARK SUMMARY ---")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
