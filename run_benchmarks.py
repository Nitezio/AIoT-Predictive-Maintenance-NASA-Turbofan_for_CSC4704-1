import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
import os
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

    # 2. Train with specific seed (Benchmarking raw performance, no buffer)
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
    
    # MULTI-CLASS F1 (Weighted)
    def categorize(rul):
        if rul < 20: return 0 # Critical
        if rul < 50: return 1 # Warning
        return 2 # Healthy

    actual_cats = actuals.apply(categorize)
    pred_cats = pd.Series(predictions).apply(categorize)
    f1_weighted = f1_score(actual_cats, pred_cats, average='weighted')
    
    # Dangerous Over-optimism
    dangerous = len(predictions[(predictions - actuals) > 20])

    return {
        "Seed": seed,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4),
        "Bias": round(bias, 2),
        "System F1": round(f1_weighted, 4),
        "Dangerous Errors": dangerous
    }

def main():
    seeds = [42, 123, 456, 789, 101112]
    results = []
    print(f"Starting 5-Trial Benchmark for High-Dependability Engine (v2.1)...")
    
    for i, s in enumerate(seeds):
        print(f"   Running Trial {i+1}/5 (Seed: {s})...")
        try:
            res = run_trial(s)
            results.append(res)
        except Exception as e:
            print(f"   Error in trial {i+1}: {e}")
    
    if not results:
        print("Error: No trials completed successfully.")
        return

    df = pd.DataFrame(results)
    
    # Calculate Averages
    avg = df.mean(numeric_only=True).to_dict()
    avg["Seed"] = "AVERAGE"
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    
    print("\n--- FINAL BENCHMARK SUMMARY ---")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))
        print("\nNote: Install 'tabulate' for better table formatting.")

if __name__ == "__main__":
    main()
