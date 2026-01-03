import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="AIoT Predictive Maintenance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# UI STYLING: Glassmorphism, Hover Effects & Color Cards
# =====================================================
st.markdown("""
<style>
    /* Main Dashboard Background */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }

    /* Sidebar Glassmorphism */
    section[data-testid="stSidebar"] {
        background-color: rgba(2, 6, 23, 0.8) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }

    /* THE HOVER FEATURE: Standard Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-10px);
        border-color: #38bdf8;
        box-shadow: 0 15px 30px rgba(56, 189, 248, 0.15);
    }

    /* FIX: FORCES FULL FORM TEXT (No more "cy..." or "...") */
    [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
    }

    [data-testid="stMetricLabel"] > div {
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    /* MAINTENANCE STATUS CARDS: Red, Yellow, Green */
    .status-card {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        transition: all 0.4s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
        color: white;
    }

    .status-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.4);
    }

    .critical-card { background-color: #ef4444; border-color: #b91c1c; }
    .warning-card { background-color: #f59e0b; border-color: #b45309; }
    .healthy-card { background-color: #10b981; border-color: #047857; }

    /* Typography */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .glow-hr {
        height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, transparent);
        border: none;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Global Visual Style for Charts
plt.style.use("dark_background")
plt.rcParams.update({
    "axes.facecolor": "none",
    "figure.facecolor": "none",
    "axes.edgecolor": "#334155",
    "grid.color": "#1e293b",
    "legend.facecolor": "#0f172a"
})


# =====================================================
# DATA LOADING
# =====================================================
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

# =====================================================
# HEADER
# =====================================================
st.markdown('<h1 class="main-title">🛡️ AIoT Command Center</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#94a3b8; font-size:1.1rem; margin-top:-5px;">NASA Turbofan Engine RUL Prediction — Group Project</p>',
    unsafe_allow_html=True)
st.markdown('<div class="glow-hr"></div>', unsafe_allow_html=True)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
with st.sidebar:
    st.markdown("### 🛰️ Navigation")
    option = st.radio(
        "Go to:",
        ["Overview & Performance", "Data Insights", "Predictions Analysis", "Unit Specific Analysis",
         "Maintenance Insights"],
        label_visibility="collapsed"
    )
    st.divider()
    st.info("System Status: **Operational**")

# =====================================================
# 1. OVERVIEW & PERFORMANCE
# =====================================================
if option == "Overview & Performance":
    st.subheader("📊 Model Performance Metrics")

    col1, col2, col3 = st.columns(3)
    mae = (preds['Actual_RUL'] - preds['Predicted_RUL']).abs().mean()
    rmse = ((preds['Actual_RUL'] - preds['Predicted_RUL']) ** 2).mean() ** 0.5

    with col1:
        st.metric("MAE (Mean Absolute Error)", f"{mae:.2f} Cycles")
    with col2:
        st.metric("RMSE (Root Mean Sq Error)", f"{rmse:.2f} Cycles")
    with col3:
        st.metric("Total Test Units", len(preds))

    st.write("###")
    st.info(
        "💡 **Model Info:** This Random Forest Regressor captures sensor degradation trends via rolling mean and standard deviation features.")

# =====================================================
# 2. DATA INSIGHTS
# =====================================================
elif option == "Data Insights":
    st.subheader("📈 Data Insights: Correlations")
    st.write("Heatmap of sensor relationships. High correlation identifies potential sensor redundancy.")

    sensor_cols = [c for c in history.columns if 'sensor' in c and '_' not in c[7:]]
    corr = history[sensor_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, cmap='coolwarm', ax=ax, center=0)
    ax.set_title("Cross-Sensor Correlation Map", color="#38bdf8", pad=20)
    st.pyplot(fig)

# =====================================================
# 3. PREDICTIONS ANALYSIS
# =====================================================
elif option == "Predictions Analysis":
    st.subheader("🔮 Actual vs Predicted RUL")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(preds['Actual_RUL'], preds['Predicted_RUL'], alpha=0.6, color='#38bdf8', s=60)
    ax.plot([0, 150], [0, 150], 'r--', label='Ideal Path')
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.legend()
    st.pyplot(fig)

    st.write("###")
    st.subheader("📋 Unit Prediction Registry")
    st.dataframe(preds, use_container_width=True)

# =====================================================
# 4. UNIT SPECIFIC ANALYSIS (LEGEND FIXED)
# =====================================================
elif option == "Unit Specific Analysis":
    st.subheader("🔍 Individual Asset Health Profile")

    selected_unit = st.selectbox("Select Unit ID to Inspect:", history['unit'].unique())
    unit_data = history[history['unit'] == selected_unit]

    sensors_to_plot = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_15']
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in sensors_to_plot:
        ax.plot(unit_data['time'], unit_data[s], label=s, linewidth=2)

    # FIX: Legend moved outside to the right to prevent overlap
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Sensors")
    ax.set_title(f"Sensor Degradation: Unit {selected_unit}", color="#38bdf8", pad=20)
    ax.set_xlabel("Time (Cycles)")
    st.pyplot(fig, bbox_inches='tight')

    unit_pred = preds[preds['unit'] == selected_unit].iloc[0]
    st.metric(f"Predicted RUL for Unit {selected_unit}", f"{unit_pred['Predicted_RUL']:.1f} Cycles")
    st.progress(min(unit_pred["Predicted_RUL"] / 150, 1.0))

# ---------------------------------------------------------
# 5. MAINTENANCE INSIGHTS (COLORED CARDS & FEATURE IMPORTANCE)
# ---------------------------------------------------------
elif option == "Maintenance Insights":
    st.subheader("🛠️ Maintenance Recommendations")

    # Threshold Logic
    critical_units = preds[preds['Predicted_RUL'] < 20]
    warning_units = preds[(preds['Predicted_RUL'] >= 20) & (preds['Predicted_RUL'] < 50)]
    healthy_units = preds[preds['Predicted_RUL'] >= 50]

    # HTML Status Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""<div class="status-card critical-card">
                        <p style="margin:0; font-size:1.1rem; opacity:0.9;">CRITICAL UNITS (<20)</p>
                        <h2 style="margin:0; font-size:3rem;">{len(critical_units)}</h2>
                        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="status-card warning-card">
                        <p style="margin:0; font-size:1.1rem; opacity:0.9;">WARNING UNITS (<50)</p>
                        <h2 style="margin:0; font-size:3rem;">{len(warning_units)}</h2>
                        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="status-card healthy-card">
                        <p style="margin:0; font-size:1.1rem; opacity:0.9;">HEALTHY UNITS</p>
                        <h2 style="margin:0; font-size:3rem;">{len(healthy_units)}</h2>
                        </div>""", unsafe_allow_html=True)

    st.write("###")
    st.subheader("Priority Fleet Status Board")


    def get_status(rul):
        if rul < 20:
            return "🔴 CRITICAL"
        elif rul < 50:
            return "🟡 WARNING"
        else:
            return "🟢 HEALTHY"


    preds_view = preds.copy()
    preds_view['Status'] = preds_view['Predicted_RUL'].apply(get_status)
    st.dataframe(preds_view.sort_values('Predicted_RUL')[['unit', 'Predicted_RUL', 'Status']], use_container_width=True)

    st.markdown("---")
    st.subheader("⚙️ Feature Importance")
    st.write("Which sensor features drove the AI's decision?")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance.head(10), x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Top 10 AI Decision Drivers", color="#38bdf8")
    st.pyplot(fig)