import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# --- CONFIGURATION ---
DATA_DIR = 'data'
SENSORS = [f'sensor_{i}' for i in range(1, 22)]
# Key sensors known to be important for Turbofan degradation
SELECTED_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_15']
WINDOW_SIZE = 5


def load_data():
    # Load raw files
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_FD001.txt'), sep=' ', header=None)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_FD001.txt'), sep=' ', header=None)
    truth_df = pd.read_csv(os.path.join(DATA_DIR, 'RUL_FD001.txt'), sep=' ', header=None)

    # Drop empty columns
    train_df.dropna(axis=1, inplace=True)
    test_df.dropna(axis=1, inplace=True)
    truth_df.dropna(axis=1, inplace=True)

    # Set headers
    cols = ['unit', 'time', 'os1', 'os2', 'os3'] + SENSORS
    train_df.columns = cols
    test_df.columns = cols

    return train_df, test_df, truth_df


def feature_engineering(df):
    df = df.copy()

    # Rolling Mean and Standard Deviation (Technical Depth: Captures trends over time)
    for col in SELECTED_SENSORS:
        df[f'{col}_mean'] = df.groupby('unit')[col].transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).mean())
        df[f'{col}_std'] = df.groupby('unit')[col].transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).std())

    return df


def prepare_rul(train_df, test_df, truth_df):
    # Calculate RUL for Training Data
    max_cycles = train_df.groupby('unit')['time'].max().reset_index()
    max_cycles.columns = ['unit', 'max']
    train_df = train_df.merge(max_cycles, on='unit', how='left')
    train_df['RUL'] = train_df['max'] - train_df['time']
    train_df.drop('max', axis=1, inplace=True)

    # Calculate RUL for Test Data (using Truth file)
    truth_df['unit'] = range(1, len(truth_df) + 1)
    truth_df.columns = ['RUL_truth', 'unit']

    # We only predict on the LAST cycle of the test data for evaluation
    test_last_cycle = test_df.groupby('unit').last().reset_index()
    test_last_cycle = test_last_cycle.merge(truth_df, on='unit', how='left')
    test_last_cycle['RUL'] = test_last_cycle['RUL_truth']  # Target for test set

    return train_df, test_last_cycle


def main():
    print("1. Loading Data...")
    train_df, test_df, truth_df = load_data()

    print("2. Feature Engineering...")
    train_processed = feature_engineering(train_df)
    test_processed = feature_engineering(test_df)

    # Get the last row of test data (with rolling features populated) for prediction
    test_last = test_processed.groupby('unit').last().reset_index()

    # Add RUL targets
    # --- FIX WAS APPLIED HERE: Passed 'test_processed' instead of 'test_df' ---
    train_final, test_final = prepare_rul(train_processed, test_processed, truth_df)

    # Define features (Raw sensors + Engineered rolling features)
    features = SELECTED_SENSORS + [f'{c}_mean' for c in SELECTED_SENSORS] + [f'{c}_std' for c in SELECTED_SENSORS]

    print("3. Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_final[features], train_final['RUL'])

    print("4. Evaluating...")
    predictions = rf.predict(test_final[features])

    # Calculate Metrics
    mae = mean_absolute_error(test_final['RUL'], predictions)
    rmse = np.sqrt(mean_squared_error(test_final['RUL'], predictions))
    r2 = r2_score(test_final['RUL'], predictions)

    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R2: {r2:.4f}")

    # Save Results for Dashboard
    results = pd.DataFrame({
        'unit': test_final['unit'],
        'Actual_RUL': test_final['RUL'],
        'Predicted_RUL': predictions
    })

    # Save feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Save entire test history for "Unit Analysis" plotting
    test_processed.to_csv('processed_test_history.csv', index=False)
    results.to_csv('predictions.csv', index=False)
    importance.to_csv('importance.csv', index=False)
    joblib.dump(rf, 'model.pkl')

    print("Done! Files saved for dashboard.")


if __name__ == "__main__":
    main()