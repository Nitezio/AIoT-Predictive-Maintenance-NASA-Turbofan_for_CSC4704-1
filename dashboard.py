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
    initial_sidebar_state="collapsed"
)

# =====================================================
# UI STYLING: Deep Navy Dark Mode with Multi-Color Glow
# =====================================================
st.markdown("""
<style>
    /* 1. Main Dashboard Background - Deep Navy */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }

    /* 2. Centered Titles */
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        text-shadow: 0px 10px 20px rgba(56, 189, 248, 0.2);
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* 3. Centering the Tabs */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 24px;
        background-color: rgba(30, 41, 59, 0.6);
        padding: 10px 20px;
        border-radius: 12px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab"] p {
        color: #94a3b8 !important;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.1) !important;
    }

    .stTabs [aria-selected="true"] p {
        color: #38bdf8 !important;
    }

    /* 4. METRIC CARDS WITH NEON GLOW HOVER */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 20px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-8px);
        border-color: #38bdf8;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.4);
    }

    /* 5. STATUS CARDS - RED / YELLOW / GREEN */
    .status-card {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 10px;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        cursor: pointer;
    }

    .status-card h2, .status-card p {
        margin: 0;
        color: white !important;
    }

    .critical-card { background-color: #ef4444; }
    .critical-card:hover { box-shadow: 0 0 30px rgba(239, 68, 68, 0.6); }

    .warning-card { background-color: #f59e0b; }
    .warning-card:hover { box-shadow: 0 0 30px rgba(245, 158, 11, 0.6); }

    .health-card { background-color: #10b981; }
    .health-card:hover { box-shadow: 0 0 30px rgba(16, 185, 129, 0.6); }

</style>
""", unsafe_allow_html=True)

# Dark theme charts
plt.style.use("dark_background")
plt.rcParams.update({
    "axes.facecolor": "#0f172a",
    "figure.facecolor": "#0f172a",
    "grid.color": "#1e293b",
    "text.color": "#f8fafc"
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
    st.error("Data files missing.")
    st.stop()

# =====================================================
# CENTERED HEADER
# =====================================================
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.markdown('<h1 class="main-title">🛡️ AIoT Command Center</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">NASA Turbofan Engine RUL Prediction — Group Project </p>',
                unsafe_allow_html=True)

with col3:
    st.write("###")
    st.success("🛰️ Status: **Operational**")

# =====================================================
# HORIZONTAL NAVIGATION
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance",
    "📈 Data Insights",
    "🔮 Predictions",
    "🔍 Unit Specific",
    "🛠️ Maintenance"
])

# ---------------------------------------------------------
# 1. PERFORMANCE
# ---------------------------------------------------------
with tab1:
    st.write("###")
    m_col1, m_col2, m_col3 = st.columns(3)
    mae = (preds['Actual_RUL'] - preds['Predicted_RUL']).abs().mean()
    rmse = ((preds['Actual_RUL'] - preds['Predicted_RUL']) ** 2).mean() ** 0.5
    m_col1.metric("Mean Absolute Error (Cycles)", f"{mae:.2f} Cycles")
    m_col2.metric("RMSE Accuracy Score", f"{rmse:.2f} Cycles")
    m_col3.metric("Monitored Engine Units", len(preds))

# ---------------------------------------------------------
# 2. DATA INSIGHTS
# ---------------------------------------------------------
with tab2:
    st.write("###")
    st.subheader("Sensor Relationship Intelligence")
    sensor_cols = [c for c in history.columns if 'sensor' in c and '_' not in c[7:]]
    corr = history[sensor_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(corr, cmap='RdYlGn', ax=ax, center=0)
    st.pyplot(fig)

# ---------------------------------------------------------
# 3. PREDICTIONS
# ---------------------------------------------------------
with tab3:
    st.write("###")
    c_p1, c_p2 = st.columns([1.5, 1])
    with c_p1:
        st.subheader("🔮 Actual vs Predicted RUL Assessment")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(preds['Actual_RUL'], preds['Predicted_RUL'], alpha=0.6, color='#38bdf8')
        ax.plot([0, 150], [0, 150], color='#ef4444', linestyle='--')
        st.pyplot(fig)
    with c_p2:
        st.subheader("📋 Unit Prediction Registry")
        st.dataframe(preds, use_container_width=True, height=400)

# ---------------------------------------------------------
# 4. UNIT SPECIFIC
# ---------------------------------------------------------
with tab4:
    st.write("###")
    selected_unit = st.selectbox("Select Asset ID:", history['unit'].unique())

    unit_data = history[history['unit'] == selected_unit]
    sensors_to_plot = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_15']
    fig, ax = plt.subplots(figsize=(16, 6))
    for s in sensors_to_plot:
        ax.plot(unit_data['time'], unit_data[s], label=s)
    ax.legend(loc='upper right')
    ax.set_title(f"Sensor Readings Over Time (Unit {selected_unit})")
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Predicted Remaining Life (Cycles)")
    unit_pred = preds[preds['unit'] == selected_unit].iloc[0]

    sum_c1, sum_c2 = st.columns([1, 3])
    with sum_c1:
        st.metric(label="Calculated RUL", value=f"{unit_pred['Predicted_RUL']:.1f}")
    with sum_c2:
        st.write("Life Consumption Progress:")
        st.progress(min(unit_pred["Predicted_RUL"] / 150, 1.0))

# ---------------------------------------------------------
# 5. MAINTENANCE
# ---------------------------------------------------------
with tab5:
    st.write("###")
    s_col1, s_col2, s_col3 = st.columns(3)

    crit = len(preds[preds['Predicted_RUL'] < 20])
    warn = len(preds[(preds['Predicted_RUL'] >= 20) & (preds['Predicted_RUL'] < 50)])
    heal = len(preds[preds['Predicted_RUL'] >= 50])

    s_col1.markdown(f'<div class="status-card critical-card"><p>CRITICAL</p><h2>{crit}</h2></div>',
                    unsafe_allow_html=True)
    s_col2.markdown(f'<div class="status-card warning-card"><p>WARNING</p><h2>{warn}</h2></div>',
                    unsafe_allow_html=True)
    s_col3.markdown(f'<div class="status-card health-card"><p>HEALTH</p><h2>{heal}</h2></div>', unsafe_allow_html=True)

    st.write("###")
    low_c1, low_c2 = st.columns([1, 1])
    with low_c1:
        st.subheader("⚙️ Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 8))
        # Updated: Changed palette to a solid white color
        sns.barplot(data=importance.head(10), x='Importance', y='Feature', color='white')
        st.pyplot(fig)
    with low_c2:
        st.subheader("Priority Status Board")
        preds_view = preds.copy()
        preds_view['Status'] = preds_view['Predicted_RUL'].apply(
            lambda x: "🔴 CRITICAL" if x < 20 else ("🟡 WARNING" if x < 50 else "🟢 HEALTH"))
        st.dataframe(preds_view.sort_values('Predicted_RUL')[['unit', 'Predicted_RUL', 'Status']],
                     use_container_width=True)  
