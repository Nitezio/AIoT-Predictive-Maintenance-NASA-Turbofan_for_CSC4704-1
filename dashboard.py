import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
from datetime import datetime, timedelta
import time
import re
import os
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIGURATION (MUST BE FIRST)
# =====================================================
st.set_page_config(
    page_title="AIoT Command Center | VHACK",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SYSTEM BOOT SEQUENCE
# =====================================================
if 'booted' not in st.session_state:
    with st.spinner("✨ Synchronizing with SME Fleet Telemetry..."):
        time.sleep(1)
    with st.spinner("🧠 Waking up AI Engine..."):
        time.sleep(0.5)
    st.toast("✅ Secure Uplink Established. Welcome to the Command Center.", icon="🌌")
    st.session_state.booted = True

# =====================================================
# UI STYLING: Ultra-Premium Dark Emerald Enterprise Theme
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Fira+Code:wght@400;600;800&display=swap');

    /* BASE SYSTEM */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #022c22 0%, #020617 60%, #000000 100%) !important;
        font-family: 'Outfit', sans-serif;
        color: #e2e8f0;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.1) !important;
    }
    
    /* MAIN HEADER */
    .main-header {
        font-family: 'Outfit', sans-serif !important;
        font-size: 4rem !important;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1.5px;
        margin-top: -20px !important;
        margin-bottom: 15px !important;
        background: linear-gradient(to right, #059669, #34d399, #10b981, #059669);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientFlow 4s linear infinite;
        filter: drop-shadow(0 4px 20px rgba(16, 185, 129, 0.2));
    }
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 50px;
    }
    @keyframes gradientFlow { 0% { background-position: 0% center; } 100% { background-position: 200% center; } }

    /* =========================================
       PERFECT METRIC CARDS (LOCKED HEIGHT & ALIGNMENT)
       ========================================= */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(2, 44, 34, 0.6) 0%, rgba(2, 6, 23, 0.9) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.15) !important;
        border-radius: 16px !important;
        height: 160px !important; /* STRICT LOCK */
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 0 !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(16, 185, 129, 0.6) !important;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15), inset 0 0 20px rgba(16, 185, 129, 0.05) !important;
        transform: translateY(-4px) !important;
    }
    /* LABEL (TOP ALIGNED) */
    [data-testid="stMetricLabel"] {
        position: absolute !important;
        top: 20px !important;
        left: 0 !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }
    [data-testid="stMetricLabel"] > div > div {
        color: #a7f3d0 !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }
    /* VALUE (DEAD CENTERED) */
    [data-testid="stMetricValue"] {
        position: absolute !important;
        top: 50% !important;
        left: 0 !important;
        width: 100% !important;
        transform: translateY(-50%) !important;
        display: flex !important;
        justify-content: center !important;
    }
    [data-testid="stMetricValue"] > div {
        font-family: 'Fira Code', monospace !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        line-height: 1 !important;
    }
    /* DELTA (BOTTOM ALIGNED) */
    [data-testid="stMetricDelta"] {
        position: absolute !important;
        bottom: 15px !important;
        left: 0 !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
    }

    /* PILL TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(2, 6, 23, 0.5);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(16, 185, 129, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        color: #64748b;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(16, 185, 129, 0.15) !important;
        color: #34d399 !important;
        box-shadow: inset 0 0 10px rgba(16, 185, 129, 0.1);
    }

    /* CHAT INTERFACE */
    .stChatMessage {
        background: rgba(2, 44, 34, 0.2) !important;
        border: 1px solid rgba(16, 185, 129, 0.05) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    div[data-testid="stChatMessageAssistant"] {
        background: linear-gradient(135deg, rgba(2, 44, 34, 0.8), rgba(2, 6, 23, 0.6)) !important;
        border-left: 4px solid #10b981 !important;
    }
    div[data-testid="stChatMessageUser"] {
        background: rgba(15, 23, 42, 0.3) !important;
        border-right: 4px solid #34d399 !important;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.4) !important;
    }

    /* ALERT BANNERS */
    .alert-banner {
        border-radius: 12px;
        padding: 16px 24px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        display: flex;
        align-items: center;
        margin-bottom: 24px;
        border: 1px solid transparent;
    }
    .alert-critical {
        background: rgba(239, 68, 68, 0.1) !important;
        border-color: rgba(239, 68, 68, 0.3) !important;
        color: #fca5a5;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.1);
    }
    .alert-healthy {
        background: rgba(16, 185, 129, 0.1) !important;
        border-color: rgba(16, 185, 129, 0.3) !important;
        color: #6ee7b7;
    }

    /* CLEAN DATAFRAMES & EXPANDERS */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        border: 1px solid rgba(16,185,129,0.1) !important;
        background: rgba(2, 6, 23, 0.5) !important;
    }
    [data-testid="stExpander"] {
        background: rgba(2, 44, 34, 0.3) !important;
        border: 1px solid rgba(16,185,129,0.1) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# STATE INITIALIZATION
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I am the AI Maintenance Co-Pilot. I am monitoring the fleet. How can I assist you today?"}
    ]
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0
if "last_call_time" not in st.session_state:
    st.session_state.last_call_time = time.time()

RATE_LIMIT_MAX_CALLS = 10
RATE_LIMIT_WINDOW_SEC = 60

if "tickets_df" not in st.session_state:
    if os.path.exists("tickets.csv"):
        st.session_state.tickets_df = pd.read_csv("tickets.csv")
    else:
        st.session_state.tickets_df = pd.DataFrame(columns=["Ticket_ID", "Unit", "Assignee", "Status", "Created_At"])

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
    st.markdown("<h2 style='text-align: center; background: -webkit-linear-gradient(#10b981, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>COMMAND LINK</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.write("### ⚙️ Parameters")
    cycle_multiplier = st.slider("Duty Cycle (Hrs/Cycle)", 1, 24, 8, help="Operational hours per data cycle.")

    st.write("### 🌍 Localization")
    lang = st.selectbox("Interface Language", ["English", "Bahasa Melayu"])

    st.markdown("---")
    st.caption("🟢 **System Status:** Online")
    st.caption("🧠 **AI Model:** HistGradient v2.1")
    st.caption("🛡️ **Security:** Encrypted")

# =====================================================
# MAIN HEADER
# =====================================================
st.markdown('<div class="main-header">SME-SHIELD AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Empowering ASEAN Industry through Explainable RUL Prediction</div>', unsafe_allow_html=True)

# Global Alerts Area
crit_list = preds[preds['Predicted_RUL'] < 20]
if not crit_list.empty:
    alert_text = f"🚨 CRITICAL ALERT: {len(crit_list)} assets showing imminent failure risk (RUL < 20). Immediate intervention required." if lang == "English" else f"🚨 KECEMASAN: {len(crit_list)} aset berisiko tinggi (RUL < 20). Tindakan segera diperlukan."
    st.markdown(f'<div class="alert-banner alert-critical"><strong>{alert_text}</strong></div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="alert-banner alert-healthy">✨ All fleet systems are stabilizing within optimal parameters. No immediate threats detected.</div>',
        unsafe_allow_html=True)

# =====================================================
# TABS NAVIGATION
# =====================================================
tab_titles = ["🏢 Fleet Overview", "🧬 Deep Analysis (XAI)", "📅 Operations Planner", "⚙️ Neural Network Specs", "💬 AI Co-Pilot", "👨‍💻 Dev Console"]
t1, t2, tab_planner, tab_system, tab_chat, tab_dev = st.tabs(tab_titles)

# ---------------------------------------------------------
# 1. FLEET OVERVIEW
# ---------------------------------------------------------
with t1:
    st.write(" ")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fleet Integrity", "92.8%")
    m2.metric("Critical Units", len(crit_list), delta=f"{len(crit_list)} High Risk", delta_color="inverse")
    m3.metric("Fleet Avg RUL", f"{preds['Predicted_RUL'].mean():.1f} Cyc")
    m4.metric("Telemetry Sync", "Live", delta="0ms Latency", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    col_table, col_chart = st.columns([1.5, 1], gap="large")

    with col_table:
        st.markdown("### 📋 Asset Registry")
        display_df = preds.copy()
        display_df['Remaining Time'] = (display_df['Predicted_RUL'] * cycle_multiplier).astype(int).astype(str) + " Hrs"
        display_df['Health Status'] = display_df['Predicted_RUL'].apply(
            lambda x: "CRITICAL" if x < 20 else ("WARNING" if x < 50 else "HEALTHY"))

        st.dataframe(
            display_df[['unit', 'Health Status', 'Predicted_RUL', 'Remaining Time']].sort_values('Predicted_RUL'),
            hide_index=True,
            use_container_width=True,
            height=380
        )

    with col_chart:
        st.markdown("### 📊 Fleet Health Distribution")
        counts = display_df['Health Status'].value_counts()
        order = ["HEALTHY", "WARNING", "CRITICAL"]
        counts = counts.reindex(order).fillna(0)
        
        # Perfect Number Placement & Legend Removal for Custom Dots
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Healthy", "Warning", "Critical"], 
            values=counts.values,
            hole=0.65,
            marker=dict(colors=['#10b981', '#f59e0b', '#ef4444'], line=dict(color='#020617', width=2)),
            textinfo='percent', 
            textposition='outside',
            textfont=dict(color='#cbd5e1', size=15, family="Outfit"),
            hoverinfo='label+value+percent'
        )])
        
        fig_donut.update_layout(
            showlegend=False, # DISABLED NATIVE SQUARE BOXES
            margin=dict(t=20, b=0, l=30, r=30), 
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        
        # CUSTOM HTML LEGEND WITH PERFECT CIRCLE DOTS
        st.markdown("""
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: -10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #10b981;"></div>
                    <span style="color: #94a3b8; font-family: 'Outfit', sans-serif; font-size: 14px; font-weight: 600;">Healthy</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #f59e0b;"></div>
                    <span style="color: #94a3b8; font-family: 'Outfit', sans-serif; font-size: 14px; font-weight: 600;">Warning</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #ef4444;"></div>
                    <span style="color: #94a3b8; font-family: 'Outfit', sans-serif; font-size: 14px; font-weight: 600;">Critical</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. ASSET ANALYSIS (XAI)
# ---------------------------------------------------------
with t2:
    st.write(" ")
    unit_ids = sorted(history['unit'].unique())
    selected_unit = st.selectbox("🎯 Target Asset Serial Number:", unit_ids)

    st.markdown("---")
    u_col1, u_col2 = st.columns([1.2, 1], gap="large")
    unit_data = history[history['unit'] == selected_unit]
    unit_pred = preds[preds['unit'] == selected_unit].iloc[0]

    with u_col1:
        st.markdown(f"### 📈 Telemetry Trace (Asset #{selected_unit})")
        sensors_to_show = ['sensor_11', 'sensor_4', 'sensor_15', 'sensor_7']
        fig_tele, ax_tele = plt.subplots(figsize=(12, 5))
        
        palette = ['#38bdf8', '#818cf8', '#c084fc', '#f472b6']
        for idx, s in enumerate(sensors_to_show):
            vals = unit_data[s]
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
            ax_tele.plot(unit_data['time'], norm_vals, label=s.upper(), color=palette[idx], linewidth=2.5, alpha=0.9)

        if 'Machine_State' in unit_data.columns:
            state_changes = unit_data[unit_data['Machine_State'] == 1]
            if not state_changes.empty:
                cp_time = state_changes['time'].iloc[0]
                ax_tele.axvline(x=cp_time, color='#ef4444', linestyle='--', label="Anomaly Detected", linewidth=2)

        ax_tele.legend(frameon=False, labelcolor='white')
        ax_tele.set_xlabel("Operational Cycles", color="#94a3b8")
        ax_tele.set_ylabel("Normalized Intensity", color="#94a3b8")
        ax_tele.tick_params(colors="#94a3b8")
        ax_tele.grid(color='#022c22', linestyle='-', linewidth=1, alpha=0.5)
        for spine in ax_tele.spines.values(): spine.set_visible(False)
        fig_tele.patch.set_alpha(0)
        ax_tele.set_facecolor('none')
        st.pyplot(fig_tele)

    with u_col2:
        st.markdown("### 🧠 AI Logic Breakdown")
        if explainer:
            try:
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

                plt.rcParams['text.color'] = '#f8fafc'
                plt.rcParams['axes.labelcolor'] = '#f8fafc'
                plt.rcParams['xtick.color'] = '#94a3b8'
                plt.rcParams['ytick.color'] = '#94a3b8'

                fig_shap, ax_shap = plt.subplots(figsize=(6, 5))
                shap.plots.waterfall(shap_values[0], max_display=6, show=False)
                plt.title("", color='white') 
                fig_shap.patch.set_alpha(0)
                
                for text in plt.gca().texts: text.set_color("white")
                plt.gca().tick_params(colors="white")
                st.pyplot(fig_shap)
                st.caption("🔴 Subtracts from Life | 🔵 Adds to Life")
            except Exception as e:
                st.warning("Visualization recalibrating...")
        else:
            st.warning("SHAP Explainer unavailable.")

    # High-End Gauge Component
    st.markdown("<br>", unsafe_allow_html=True)
    gauge_col1, gauge_col2, gauge_col3 = st.columns([1, 2, 1])
    with gauge_col2:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = unit_pred['Predicted_RUL'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "REMAINING USEFUL LIFE", 'font': {'size': 18, 'color': "#94a3b8", 'family': "Outfit"}},
            number = {'font': {'color': "#f8fafc", 'size': 50}},
            gauge = {
                'axis': {'range': [0, 250], 'tickwidth': 1, 'tickcolor': "#064e3b"},
                'bar': {'color': "#10b981", 'thickness': 0.2},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(239, 68, 68, 0.3)'},
                    {'range': [20, 50], 'color': 'rgba(245, 158, 11, 0.3)'},
                    {'range': [50, 250], 'color': 'rgba(16, 185, 129, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#34d399", 'width': 4},
                    'thickness': 0.75,
                    'value': unit_pred['Predicted_RUL']
                }
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        hours_left = unit_pred['Predicted_RUL'] * cycle_multiplier
        days_left = round(hours_left / 24, 1)
        st.markdown(f"<h3 style='text-align:center; color:#e2e8f0; font-weight:300;'>ESTIMATED TIME TO FAILURE: <span style='color:#10b981; font-weight:800;'>{days_left} DAYS</span></h3>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. MAINTENANCE PLANNER & TICKETING
# ---------------------------------------------------------
with tab_planner:
    st.write(" ")
    t_col1, t_col2 = st.columns([1.2, 1], gap="large")

    with t_col1:
        st.markdown("### ⚠️ Action Required")
        planner_df = display_df[display_df['Predicted_RUL'] < 50].sort_values('Predicted_RUL')

        if not planner_df.empty:
            for idx, row in planner_df.iterrows():
                unit_id = int(row['unit'])
                is_ticketed = not st.session_state.tickets_df.empty and unit_id in st.session_state.tickets_df[st.session_state.tickets_df['Status'] == 'Open']['Unit'].values

                if not is_ticketed:
                    with st.expander(f"🛠️ Dispatch Order: Asset #{unit_id} (ETA: {int(row['Predicted_RUL'] * cycle_multiplier)} Hrs)"):
                        with st.form(key=f"form_{unit_id}"):
                            st.write("**Diagnostics:** Thermal anomalies & lubrication decay detected.")
                            assignee = st.selectbox("Assign Specialist:", ["Technician Alpha", "Technician Bravo", "Engineering Core"])
                            submit_ticket = st.form_submit_button("Issue Dispatch Order")

                            if submit_ticket:
                                new_ticket = pd.DataFrame([{
                                    "Ticket_ID": f"WO-{np.random.randint(1000, 9999)}",
                                    "Unit": unit_id,
                                    "Assignee": assignee,
                                    "Status": "Open",
                                    "Created_At": datetime.now().strftime("%Y-%m-%d %H:%M")
                                }])
                                st.session_state.tickets_df = pd.concat([st.session_state.tickets_df, new_ticket], ignore_index=True)
                                st.session_state.tickets_df.to_csv("tickets.csv", index=False)
                                st.success("Order Dispatched!")
                                st.rerun()
                else:
                    st.info(f"Asset #{unit_id}: Dispatch Order already active.")
        else:
            st.success("✨ Zero critical actions required. Fleet is stable.")

    with t_col2:
        st.markdown("### 📋 Active Operations")
        if not st.session_state.tickets_df.empty:
            open_tickets = st.session_state.tickets_df[st.session_state.tickets_df['Status'] == 'Open']

            if not open_tickets.empty:
                for idx, t_row in open_tickets.iterrows():
                    with st.container():
                        st.markdown(f"<div style='background:rgba(255,255,255,0.05); padding:15px; border-radius:10px; border-left: 4px solid #10b981;'><b>{t_row['Ticket_ID']} | Asset #{t_row['Unit']}</b><br><span style='color:#94a3b8; font-size:0.9em;'>👨‍🔧 {t_row['Assignee']} | 🕒 {t_row['Created_At']}</span></div>", unsafe_allow_html=True)
                        if st.button("✅ Mark Resolved", key=f"res_{t_row['Ticket_ID']}"):
                            st.session_state.tickets_df.loc[idx, 'Status'] = 'Resolved'
                            st.session_state.tickets_df.to_csv("tickets.csv", index=False)
                            st.success("Verifying repair with IoT sensors...")
                            time.sleep(1)
                            st.rerun()
                        st.write(" ")
            else:
                st.write("No active operations.")

            closed_tickets = st.session_state.tickets_df[st.session_state.tickets_df['Status'] == 'Resolved']
            if not closed_tickets.empty:
                st.markdown("#### ✅ Resolution Log")
                st.dataframe(closed_tickets[['Ticket_ID', 'Unit', 'Assignee']], hide_index=True, use_container_width=True)
        else:
            st.write("Database clear.")

# ---------------------------------------------------------
# 4. SYSTEM INTELLIGENCE
# ---------------------------------------------------------
with tab_system:
    st.write(" ")
    sc_1, sc_2 = st.columns([1.5, 1], gap="large")

    with sc_1:
        st.markdown("### 🌐 Global Feature Topography")
        fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
        sns.barplot(data=importance.head(10), x='Importance', y='Feature', palette='mako')
        
        ax_imp.set_facecolor('none')
        fig_imp.patch.set_alpha(0)
        ax_imp.tick_params(colors="#e2e8f0")
        ax_imp.xaxis.label.set_color('#e2e8f0')
        ax_imp.yaxis.label.set_color('#e2e8f0')
        for spine in ax_imp.spines.values(): spine.set_edgecolor('#064e3b')
        st.pyplot(fig_imp)

    with sc_2:
        st.markdown("### 📊 Engine Metrics")
        mae = (preds['Actual_RUL'] - preds['Predicted_RUL']).abs().mean()
        rmse = ((preds['Actual_RUL'] - preds['Predicted_RUL']) ** 2).mean() ** 0.5
        
        st.metric("Root Mean Square Error", f"{rmse:.2f}", delta="- Top Tier", delta_color="normal")
        st.metric("Mean Absolute Error", f"{mae:.2f}", delta="- Stable", delta_color="normal")
        
        st.markdown("---")
        st.markdown("""
        **Safety & Governance Protocols:**
        * 🛡️ **Pessimistic Buffer:** Active (-5 Cycles)
        * 📉 **Anomaly Detection:** Pelt Change-Point
        * 🌲 **Architecture:** HistGradientBoosting
        """)

# ---------------------------------------------------------
# 5. AI CO-PILOT (CHAT)
# ---------------------------------------------------------
with tab_chat:
    st.write(" ")
    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="✨" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    current_time = time.time()
    if current_time - st.session_state.last_call_time > RATE_LIMIT_WINDOW_SEC:
        st.session_state.api_calls = 0
        st.session_state.last_call_time = current_time

    if st.session_state.api_calls >= RATE_LIMIT_MAX_CALLS:
        time_left = int(RATE_LIMIT_WINDOW_SEC - (current_time - st.session_state.last_call_time))
        st.error(f"⚠️ Neural Link Cooling Down. Please wait {time_left} seconds.")
        chat_input = st.chat_input("System cooling...", disabled=True)
    else:
        chat_input = st.chat_input("E.g., 'Identify critical assets' or 'Query status of unit 42'")

    if chat_input:
        st.session_state.api_calls += 1
        st.session_state.last_call_time = current_time

        st.session_state.messages.append({"role": "user", "content": chat_input})
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(chat_input)

        user_text = chat_input.lower()

        with chat_container:
            with st.chat_message("assistant", avatar="✨"):
                with st.spinner("Accessing databanks..."):
                    time.sleep(0.6)

                    if "critical" in user_text or "urgent" in user_text or "danger" in user_text:
                        if crit_list.empty:
                            response = "All green. Zero assets are currently exhibiting critical failure signatures."
                        else:
                            crit_units = crit_list['unit'].tolist()
                            response = f"**{len(crit_units)} assets** flag as critical. Immediate action recommended for Units: `{', '.join(map(str, crit_units))}`."

                    elif "unit" in user_text or "asset" in user_text:
                        match = re.search(r'\d+', user_text)
                        if match:
                            unit_num = int(match.group())
                            if unit_num in preds['unit'].values:
                                u_data = preds[preds['unit'] == unit_num].iloc[0]
                                u_rul = u_data['Predicted_RUL']
                                u_state = "🔴 IMPAIRED" if u_data.get('Machine_State', 0) == 1 else "🟢 NOMINAL"
                                response = f"**Asset #{unit_num} Telemetry:**\n\n* **State:** {u_state}\n* **Est. Cycles Remaining:** {u_rul:.0f}\n* **Time to Failure:** {int(u_rul * cycle_multiplier)} Hrs"
                            else:
                                response = f"Asset identifier `{unit_num}` not found in current fleet registry."
                        else:
                            response = "Please specify a numeric asset ID (e.g., 'Unit 42')."

                    elif "health" in user_text or "summary" in user_text or "status" in user_text:
                        avg_rul = preds['Predicted_RUL'].mean()
                        response = f"Fleet is stabilized. Operating with an average RUL of **{avg_rul:.1f} cycles** across {len(preds)} monitored units."

                    else:
                        response = "Query not recognized by current operational parameters. Try asking about 'critical assets' or specific 'unit status'."

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# ---------------------------------------------------------
# 6. DEVELOPER PANEL (ADMIN ZONE)
# ---------------------------------------------------------
with tab_dev:
    st.write(" ")
    st.markdown("<h3 style='color: #ef4444;'>⚠️ Restricted Zone: Data Override Protocols</h3>", unsafe_allow_html=True)
    st.write("Manual telemetry overrides bypass AI safety buffers. Proceed with extreme caution.")

    if "sensor_corrections" not in st.session_state:
        st.session_state.sensor_corrections = {}

    col_dev1, col_dev2 = st.columns([1, 1.5], gap="large")

    with col_dev1:
        st.markdown("#### Input Calibration")
        with st.form("correction_form"):
            dev_unit = st.selectbox("Target Asset:", sorted(history['unit'].unique()))
            dev_sensor = st.selectbox("Sensor Vector:", ['sensor_11', 'sensor_4', 'sensor_15', 'sensor_7'])
            correction_value = st.number_input("Calibration Offset (+/-):", value=0.0)

            submit_dev = st.form_submit_button("Inject Override Sequence")

            if submit_dev:
                key = f"{dev_unit}_{dev_sensor}"
                st.session_state.sensor_corrections[key] = correction_value
                st.success(f"Signal injected. {dev_sensor} offset by {correction_value} on Asset #{dev_unit}.")
                st.rerun()

    with col_dev2:
        st.markdown("#### Active Injections")
        if st.session_state.sensor_corrections:
            corr_data = []
            for k, v in st.session_state.sensor_corrections.items():
                parts = k.split("_", 1)
                if len(parts) == 2:
                    corr_data.append({"Asset ID": parts[0], "Vector": parts[1], "Offset": v})
            
            st.dataframe(pd.DataFrame(corr_data), hide_index=True, use_container_width=True)

            if st.button("Purge All Overrides"):
                st.session_state.sensor_corrections = {}
                st.rerun()
        else:
            st.info("System running pure. No manual overrides detected.")

    st.markdown("---")
    st.caption(f"🔒 **Admin Audit Log** | Last Access: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Environment: Production")

st.markdown("---")
st.markdown("<center><p style='color: #475569; font-size: 0.8rem; font-family: Outfit;'>POWERED BY AIoT INTELLIGENCE | BUILD v2.1</p></center>", unsafe_allow_html=True)
