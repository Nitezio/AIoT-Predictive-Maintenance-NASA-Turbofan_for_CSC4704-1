import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Page Config
st.set_page_config(page_title="AIoT Predictive Maintenance", layout="wide")


# --- LOAD DATA ---
@st.cache_data
def load_dashboard_data():
    preds = pd.read_csv('predictions.csv')
    importance = pd.read_csv('importance.csv')
    history = pd.read_csv('processed_test_history.csv')
    return preds, importance, history


try:
    preds, importance, history = load_dashboard_data()
except FileNotFoundError:
    st.error("Please run 'python train_model.py' first to generate results!")
    st.stop()

# --- HEADER ---
st.title("🛡️ AIoT Predictive Maintenance System")
st.markdown("**Group Project:** NASA Turbofan Engine RUL Prediction")
st.markdown("---")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:",
                          ["Overview & Performance", "Data Insights", "Predictions Analysis", "Unit Specific Analysis",
                           "Maintenance Insights"])

# ---------------------------------------------------------
# 1. OVERVIEW & MODEL PERFORMANCE
# ---------------------------------------------------------
if option == "Overview & Performance":
    st.header("📊 Model Performance Metrics")

    col1, col2, col3 = st.columns(3)

    mae = (preds['Actual_RUL'] - preds['Predicted_RUL']).abs().mean()
    rmse = ((preds['Actual_RUL'] - preds['Predicted_RUL']) ** 2).mean() ** 0.5

    col1.metric("MAE (Mean Absolute Error)", f"{mae:.2f} cycles")
    col2.metric("RMSE (Root Mean Sq Error)", f"{rmse:.2f} cycles")
    col3.metric("Total Test Units", len(preds))

    st.info(
        "The model uses a Random Forest Regressor trained on 100 engine units. It utilizes rolling mean and standard deviation features to capture sensor degradation trends.")

# ---------------------------------------------------------
# 2. DATA INSIGHTS
# ---------------------------------------------------------
elif option == "Data Insights":
    st.header("📈 Data Insights: Correlations")

    st.write(
        "Correlation heatmap shows which sensors move together. High correlation implies redundancy or strong relationships.")

    # Filter for sensor columns only
    sensor_cols = [c for c in history.columns if 'sensor' in c and '_' not in c[7:]]
    corr = history[sensor_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ---------------------------------------------------------
# 3. PREDICTIONS ANALYSIS
# ---------------------------------------------------------
elif option == "Predictions Analysis":
    st.header("🔮 Actual vs Predicted RUL")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(preds['Actual_RUL'], preds['Predicted_RUL'], alpha=0.6, color='blue')
    ax.plot([0, 150], [0, 150], 'r--', label='Perfect Prediction')  # Ideal line
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.dataframe(preds.head(10))

# ---------------------------------------------------------
# 4. UNIT SPECIFIC ANALYSIS
# ---------------------------------------------------------
elif option == "Unit Specific Analysis":
    st.header("🔍 Unit-Specific Sensor History")

    selected_unit = st.selectbox("Select Unit ID to Inspect:", history['unit'].unique())

    # Filter data for this unit
    unit_data = history[history['unit'] == selected_unit]

    st.subheader(f"Sensor Degradation: Unit {selected_unit}")

    # Plot key sensors
    sensors_to_plot = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_15']
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in sensors_to_plot:
        ax.plot(unit_data['time'], unit_data[s], label=s)

    ax.set_title("Key Sensors over Time")
    ax.set_xlabel("Time (Cycles)")
    ax.set_ylabel("Sensor Value (Normalized/Raw)")
    ax.legend()
    st.pyplot(fig)

    # Show prediction for this unit
    unit_pred = preds[preds['unit'] == selected_unit].iloc[0]
    st.metric(f"Final Predicted RUL for Unit {selected_unit}", f"{unit_pred['Predicted_RUL']:.2f} cycles")

# ---------------------------------------------------------
# 5. MAINTENANCE INSIGHTS (Most Important Tab)
# ---------------------------------------------------------
elif option == "Maintenance Insights":
    st.header("🛠️ Maintenance Recommendations")


    # Color Coded Status
    def get_status(rul):
        if rul < 20:
            return "CRITICAL 🚨"
        elif rul < 50:
            return "WARNING ⚠️"
        else:
            return "HEALTHY ✅"


    preds['Status'] = preds['Predicted_RUL'].apply(get_status)

    # Show Critical Units First
    st.subheader("Fleet Status Board")

    # Counters
    col1, col2, col3 = st.columns(3)
    critical_count = len(preds[preds['Status'].str.contains("CRITICAL")])
    warning_count = len(preds[preds['Status'].str.contains("WARNING")])

    col1.error(f"Critical Units: {critical_count}")
    col2.warning(f"Warning Units: {warning_count}")
    col3.success(f"Healthy Units: {len(preds) - critical_count - warning_count}")

    # Filterable Table
    st.dataframe(preds.sort_values('Predicted_RUL')[['unit', 'Predicted_RUL', 'Status']], use_container_width=True)

    st.markdown("---")
    st.subheader("⚙️ Feature Importance")
    st.write("Which sensor features drove the AI's decision?")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance.head(10), x='Importance', y='Feature', palette='viridis', ax=ax)
    st.pyplot(fig)