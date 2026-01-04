import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

DATA_DIR = 'data'
# FD001 relevant sensors (excluding flat-line sensors)
SELECTED_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
                    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
                    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

# Standard threshold for FD001 where degradation actually becomes visible
RUL_CLIP_LIMIT = 125


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_FD001.txt'), sep=' ', header=None)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_FD001.txt'), sep=' ', header=None)
    truth_df = pd.read_csv(os.path.join(DATA_DIR, 'RUL_FD001.txt'), sep=' ', header=None)

    train_df.dropna(axis=1, inplace=True)
    test_df.dropna(axis=1, inplace=True)
    truth_df.dropna(axis=1, inplace=True)

    # FD001 has 21 sensors + 3 settings
    cols = ['unit', 'time', 'os1', 'os2', 'os3'] + [f'sensor_{i}' for i in range(1, 22)]
    train_df.columns = cols
    test_df.columns = cols
    return train_df, test_df, truth_df


def feature_engineering(df):
    df = df.copy()

    # --- FIX 1: REMOVED UNIT-WISE NORMALIZATION ---
    # Random Forest works well with raw values.
    # If you want to normalize, fit a StandardScaler on TRAIN and transform TEST globally.
    # For now, raw values preserve the degradation signal better than unit-wise scaling.

    # Multi-scale rolling degradation features
    # (Computed on RAW sensor values now, so the absolute magnitude is preserved)
    for w in [5, 10, 20]:
        for s in SELECTED_SENSORS:
            # Rolling Mean
            df[f'{s}_mean_{w}'] = df.groupby('unit')[s].rolling(w, min_periods=1).mean().reset_index(0, drop=True)
            # Rolling Std (captures increasing volatility)
            df[f'{s}_std_{w}'] = df.groupby('unit')[s].rolling(w, min_periods=1).std().reset_index(0, drop=True)

    # Fill any NaNs created by rolling std with 0
    df.fillna(0, inplace=True)
    return df


def prepare_train_data(train_df):
    # Calculate Max Cycles per unit
    max_cycles = train_df.groupby('unit')['time'].max().reset_index()
    max_cycles.columns = ['unit', 'max']

    train_df = train_df.merge(max_cycles, on='unit', how='left')

    # Calculate Linear RUL
    train_df['RUL'] = train_df['max'] - train_df['time']

    # --- FIX 2: CLIP RUL ---
    # We cap the RUL. If RUL > 125, we treat it as 125.
    # This helps the model focus on the degradation phase, not the healthy phase.
    train_df['RUL'] = train_df['RUL'].clip(upper=RUL_CLIP_LIMIT)

    train_df.drop('max', axis=1, inplace=True)
    return train_df


def prepare_test_data(test_df, truth_df):
    # We only predict the LAST cycle of the test data (standard CMAPSS protocol)
    truth_df['unit'] = range(1, len(truth_df) + 1)
    truth_df.columns = ['RUL_truth', 'unit']

    # Get the last row of features for every unit
    test_last_cycle = test_df.groupby('unit').last().reset_index()

    # Attach the Ground Truth
    test_last_cycle = test_last_cycle.merge(truth_df, on='unit', how='left')

    # For testing, RUL is just the truth value (we clip it for metric comparison consistency if needed,
    # but usually we compare against true RUL. Here we clip prediction, not truth).
    test_last_cycle['RUL'] = test_last_cycle['RUL_truth']

    # Filter out units that ran longer than RUL_CLIP_LIMIT if we want strict comparison,
    # but for standard RMSE, we keep them.
    return test_last_cycle


def main():
    print("1. Loading Data...")
    train_df, test_df, truth_df = load_data()

    print("2. Feature Engineering...")
    # Generate features
    train_processed = feature_engineering(train_df)
    test_processed = feature_engineering(test_df)

    # Prepare Targets
    train_final = prepare_train_data(train_processed)
    test_final = prepare_test_data(test_processed, truth_df)

    # Define Feature List
    features = []
    # Add raw sensors + engineered features
    features.extend(SELECTED_SENSORS)
    for w in [5, 10, 20]:
        for c in SELECTED_SENSORS:
            features.append(f'{c}_mean_{w}')
            features.append(f'{c}_std_{w}')

    print(f"3. Training Random Forest on {len(features)} features...")
    rf = RandomForestRegressor(
        n_estimators=100,  # Reduced for speed, 250 is also fine
        max_depth=12,  # Prevent overfitting
        min_samples_leaf=5,  # Generalization
        random_state=42,
        n_jobs=-1
    )
    rf.fit(train_final[features], train_final['RUL'])

    print("4. Evaluating...")
    predictions = rf.predict(test_final[features])

    # Evaluation Metrics
    mae = mean_absolute_error(test_final['RUL'], predictions)
    rmse = np.sqrt(mean_squared_error(test_final['RUL'], predictions))
    r2 = r2_score(test_final['RUL'], predictions)

    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R2: {r2:.4f}")

    # --- SAVE OUTPUTS ---
    results = pd.DataFrame({
        'unit': test_final['unit'],
        'Actual_RUL': test_final['RUL'],
        'Predicted_RUL': predictions
    })

    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    results.to_csv('predictions.csv', index=False)
    importance.to_csv('importance.csv', index=False)
    joblib.dump(rf, 'model.pkl')

    print("Done! Model corrected.")


if __name__ == "__main__":
    main()
