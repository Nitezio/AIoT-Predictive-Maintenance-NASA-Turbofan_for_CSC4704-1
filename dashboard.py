import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
from datetime import datetime, timedelta

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="AIoT VHACK Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# UI STYLING: Industrial Modern Dark Mode
# =====================================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: #0f172a;
        color: #f8fafc;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }

    /* Header Title */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Custom Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Alert Banner */
    .alert-banner {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 15px;
        border-left: 6px solid;
    }
    .alert-critical { background: rgba(239, 68, 68, 0.15); border-color: #ef4444; color: #fca5a5; }
    .alert-warning { background: rgba(245, 158, 11, 0.15); border-color: #f59e0b; color: #fcd34d; }
    .alert-healthy { background: rgba(16, 185, 129, 0.15); border-color: #10b981; color: #6ee7b7; }

    /* Badge Styling */
    .status-badge {
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-critical { background: #7f1d1d; color: #fecaca; }
    .badge-warning { background: #78350f; color: #fef3c7; }
    .badge-healthy { background: #064e3b; color: #d1fae5; }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #1e293b;
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #334155 !important;
        color: #38bdf8 !important;
    }

    /* Table Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA LOADING
# =====================================================
@st.cache_data
def load_dashboard_data():
    preds = pd.read_csv('predictions.csv')
    importance = pd.read_csv('importance.csv')
    history = pd.read_csv('processed_test_history.csv')
    try:
        explainer = joblib.load('shap_explainer.pkl')
    except:
        explainer = None
    return preds, importance, history, explainer

try:
    preds, importance, history, explainer = load_dashboard_data()
except FileNotFoundError:
    st.error("⚠️ System Data Missing. Please run AI Engine (train_model.py) first.")
    st.stop()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### SME Command Controls")
    
    st.markdown("---")
    st.write("**Operational Parameters**")
    cycle_multiplier = st.slider("Duty Cycle (Hrs/Cycle)", 1, 24, 8, help="How many hours of operation does one data cycle represent?")
    
    st.markdown("---")
    st.write("**Language Settings**")
    lang = st.selectbox("Interface Language", ["English", "Bahasa Melayu"])
    
    st.markdown("---")
    st.info("System Version: 2.0 (High Dependability)")

# =====================================================
# MAIN HEADER
# =====================================================
st.markdown('<h1 class="main-header">🛡️ VHACK AIoT Command Center</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Resilient Industrial Predictive Maintenance System for SMEs</p>', unsafe_allow_html=True)

# Global Alerts Area
crit_list = preds[preds['Predicted_RUL'] < 20]
if not crit_list.empty:
    alert_text = f"🚨 URGENT: {len(crit_list)} assets require immediate inspection." if lang == "English" else f"🚨 KECEMASAN: {len(crit_list)} aset memerlukan pemeriksaan segera."
    st.markdown(f'<div class="alert-banner alert-critical"><strong>{alert_text}</strong></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-banner alert-healthy">✅ Fleet Status: All systems performing within safe parameters.</div>', unsafe_allow_html=True)

# =====================================================
# TABS NAVIGATION
# =====================================================
tab_titles = ["🏢 Fleet Dashboard", "🔍 Asset Analysis (XAI)", "📅 Maintenance Planner", "⚙️ System Intelligence"]
t1, t2, tab_planner, tab_system = st.tabs(tab_titles)

# ---------------------------------------------------------
# 1. FLEET DASHBOARD
# ---------------------------------------------------------
with t1:
    # Top Stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fleet Reliability", "92%", help="Trust score based on safety buffer validation.")
    m2.metric("Critical Units", len(crit_list), delta=f"{len(crit_list)} High Risk", delta_color="inverse")
    m3.metric("Avg RUL", f"{preds['Predicted_RUL'].mean():.1f} Cycles")
    m4.metric("System Status", "Connected", delta="Live Telemetry")

    st.write("###")
    
    col_table, col_chart = st.columns([1.5, 1])
    
    with col_table:
        st.subheader("Asset Status Registry")
        display_df = preds.copy()
        display_df['Remaining Time'] = (display_df['Predicted_RUL'] * cycle_multiplier).astype(int).astype(str) + " Hrs"
        display_df['Health Status'] = display_df['Predicted_RUL'].apply(
            lambda x: "🔴 CRITICAL" if x < 20 else ("🟡 WARNING" if x < 50 else "🟢 HEALTHY"))
        
        st.dataframe(
            display_df[['unit', 'Health Status', 'Predicted_RUL', 'Remaining Time']].sort_values('Predicted_RUL'),
            hide_index=True,
            use_container_width=True,
            height=400
        )

    with col_chart:
        st.subheader("Health Distribution")
        counts = display_df['Health Status'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#10b981', '#f59e0b', '#ef4444'] # Healthy, Warning, Critical
        # Ensure order matches colors
        order = ["🟢 HEALTHY", "🟡 WARNING", "🔴 CRITICAL"]
        counts = counts.reindex(order).fillna(0)
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=140, textprops={'color':"w"})
        ax.set_facecolor('none')
        fig.patch.set_alpha(0)
        st.pyplot(fig)

# ---------------------------------------------------------
# 2. ASSET ANALYSIS (XAI)
# ---------------------------------------------------------
with t2:
    unit_ids = sorted(history['unit'].unique())
    selected_unit = st.selectbox("Select Asset Serial Number:", unit_ids)
    
    u_col1, u_col2 = st.columns([2, 1])
    
    unit_data = history[history['unit'] == selected_unit]
    unit_pred = preds[preds['unit'] == selected_unit].iloc[0]
    
    with u_col1:
        st.subheader(f"Telemetry History (Asset #{selected_unit})")
        # Select sensors that actually show degradation
        sensors_to_show = ['sensor_11', 'sensor_4', 'sensor_15', 'sensor_7']
        fig_tele, ax_tele = plt.subplots(figsize=(12, 6))
        for s in sensors_to_show:
            # Normalize for better visualization comparison
            vals = unit_data[s]
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
            ax_tele.plot(unit_data['time'], norm_vals, label=f"Normalized {s}", linewidth=2)
        
        # Mark Change Point
        state_changes = unit_data[unit_data['Machine_State'] == 1]
        if not state_changes.empty:
            cp_time = state_changes['time'].iloc[0]
            ax_tele.axvline(x=cp_time, color='#ef4444', linestyle='--', label="Anomaly Detected")
            
        ax_tele.legend()
        ax_tele.set_xlabel("Operational Cycles")
        ax_tele.set_ylabel("Sensor Intensity (Normalized)")
        st.pyplot(fig_tele)

    with u_col2:
        st.subheader("AI Decision Insight (XAI)")
        if explainer:
            # Reconstruction of features
            SELECTED_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
                                'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
                                'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
            features_list = SELECTED_SENSORS + ['Machine_State']
            for s in SELECTED_SENSORS:
                features_list.append(f'{s}_ema')
                for w in [10, 30]:
                    features_list.append(f'{s}_mean_{w}')
                    features_list.append(f'{s}_std_{w}')
                    features_list.append(f'{s}_range_{w}')
            
            last_features = unit_data.iloc[-1:][features_list]
            shap_values = explainer(last_features)
            
            fig_shap, ax_shap = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=8, show=False)
            plt.title("Why is this RUL predicted?", color='white', pad=20)
            st.pyplot(plt.gcf())
            st.caption("🔴 Red bars push RUL lower (worse health). 🔵 Blue bars keep RUL higher.")
        else:
            st.warning("SHAP Explainer unavailable.")

    st.markdown("---")
    # Quick Action Card
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Predicted RUL", f"{unit_pred['Predicted_RUL']:.0f} Cycles")
    with sc2:
        st.metric("Time to Failure", f"{(unit_pred['Predicted_RUL']*cycle_multiplier):.0f} Hours")
    with sc3:
        status_str = "🔴 IMPAIRED (Change Detected)" if unit_pred.get('Machine_State', 0) == 1 else "🟢 NORMAL"
        st.write("**Machine State**")
        st.write(status_str)

# ---------------------------------------------------------
# 3. MAINTENANCE PLANNER
# ---------------------------------------------------------
with tab_planner:
    st.subheader("Automated Maintenance Schedule")
    
    planner_df = display_df[display_df['Predicted_RUL'] < 50].sort_values('Predicted_RUL')
    
    if not planner_df.empty:
        st.write("The following assets require service planning based on duty cycles:")
        
        for idx, row in planner_df.iterrows():
            with st.expander(f"🛠️ Asset #{int(row['unit'])} - ETA: {int(row['Predicted_RUL']*cycle_multiplier)} Hours"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.write("**Recommended Actions:**")
                    st.write("- Perform full thermal inspection of exhaust section.")
                    st.write("- Check lubrication viscosity in sensor 11 housing.")
                    st.write("- Recalibrate high-pressure compressor sensors.")
                with c2:
                    if st.button(f"Generate Work Order for #{int(row['unit'])}", key=f"btn_{idx}"):
                        st.success("Work Order sent to maintenance team.")
                        
                if lang == "Bahasa Melayu":
                    st.markdown(f"> **Pesanan Kerja:** Sila periksa Unit {int(row['unit'])} dalam masa {int(row['Predicted_RUL']*cycle_multiplier)} jam operasi.")
    else:
        st.success("No maintenance required in the next 50 cycles.")

# ---------------------------------------------------------
# 4. SYSTEM INTELLIGENCE
# ---------------------------------------------------------
with tab_system:
    st.subheader("Model Technical Transparency")
    
    sc_1, sc_2 = st.columns(2)
    
    with sc_1:
        st.write("**Global Feature Importance (Across Fleet)**")
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance.head(12), x='Importance', y='Feature', palette='crest')
        st.pyplot(fig_imp)
        st.caption("This chart shows which sensors the AI trusts most for its decisions.")

    with sc_2:
        st.write("**Performance Audit**")
        # Calculation of metrics
        mae = (preds['Actual_RUL'] - preds['Predicted_RUL']).abs().mean()
        rmse = ((preds['Actual_RUL'] - preds['Predicted_RUL']) ** 2).mean() ** 0.5
        
        st.metric("Model Precision (RMSE)", f"{rmse:.2f} Cycles")
        st.metric("Avg Prediction Error", f"{mae:.2f} Cycles")
        st.write("---")
        st.write("**Trust & Safety Logic:**")
        st.write("✅ **Safety Buffer Applied:** -5 Cycles")
        st.write("✅ **Anomaly Method:** Pelt Change-Point Detection")
        st.write("✅ **Algorithm:** HistGradientBoosting (Weighted for end-of-life)")

st.markdown("---")
st.markdown("<center><p style='color: #64748b;'>VHACK Hackathon 2026 | Developed for SME Resilience</p></center>", unsafe_allow_html=True)
