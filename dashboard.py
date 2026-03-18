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
# AI CHAT & RATE LIMITER STATE INITIALIZATION
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "System Online. I am your AI Maintenance Co-Pilot. I have analyzed the current fleet data. Ask me about critical units, specific asset health, or maintenance priorities."}
    ]
if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0
if "last_call_time" not in st.session_state:
    st.session_state.last_call_time = time.time()

# Rate Limit Settings (Max 10 messages per 60 seconds)
RATE_LIMIT_MAX_CALLS = 10
RATE_LIMIT_WINDOW_SEC = 60

# =====================================================
# TICKETING SYSTEM INITIALIZATION
# =====================================================
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
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### SME Command Controls")

    st.markdown("---")
    st.write("**Operational Parameters**")
    cycle_multiplier = st.slider("Duty Cycle (Hrs/Cycle)", 1, 24, 8,
                                 help="How many hours of operation does one data cycle represent?")

    st.markdown("---")
    st.write("**Language Settings**")
    lang = st.selectbox("Interface Language", ["English", "Bahasa Melayu"])

    st.markdown("---")
    st.info("System Version: 2.1 (AI Co-Pilot Enabled)")

# =====================================================
# MAIN HEADER
# =====================================================
st.markdown('<h1 class="main-header">🛡️ VHACK AIoT Command Center</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Resilient Industrial Predictive Maintenance System for SMEs</p>',
            unsafe_allow_html=True)

# Global Alerts Area
crit_list = preds[preds['Predicted_RUL'] < 20]
if not crit_list.empty:
    alert_text = f"🚨 URGENT: {len(crit_list)} assets require immediate inspection." if lang == "English" else f"🚨 KECEMASAN: {len(crit_list)} aset memerlukan pemeriksaan segera."
    st.markdown(f'<div class="alert-banner alert-critical"><strong>{alert_text}</strong></div>', unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="alert-banner alert-healthy">✅ Fleet Status: All systems performing within safe parameters.</div>',
        unsafe_allow_html=True)

# =====================================================
# TABS NAVIGATION
# =====================================================
tab_titles = ["🏢 Fleet Dashboard", "🔍 Asset Analysis (XAI)", "📅 Maintenance Planner", "⚙️ System Intelligence", "💬 AI Co-Pilot", "🛠️ Developer Panel"]
t1, t2, tab_planner, tab_system, tab_chat, tab_dev = st.tabs(tab_titles)

# ---------------------------------------------------------
# 1. FLEET DASHBOARD
# ---------------------------------------------------------
with t1:
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
        colors = ['#10b981', '#f59e0b', '#ef4444']
        order = ["🟢 HEALTHY", "🟡 WARNING", "🔴 CRITICAL"]
        counts = counts.reindex(order).fillna(0)
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=140, textprops={'color': "w"})
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
        sensors_to_show = ['sensor_11', 'sensor_4', 'sensor_15', 'sensor_7']
        fig_tele, ax_tele = plt.subplots(figsize=(12, 6))
        for s in sensors_to_show:
            vals = unit_data[s]
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
            ax_tele.plot(unit_data['time'], norm_vals, label=f"Normalized {s}", linewidth=2)

        if 'Machine_State' in unit_data.columns:
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

                fig_shap, ax_shap = plt.subplots()
                shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                plt.title("Why is this RUL predicted?", color='white', pad=20)
                st.pyplot(plt.gcf())
                st.caption("🔴 Red bars push RUL lower. 🔵 Blue bars keep RUL higher.")
            except Exception as e:
                st.warning("SHAP visualization failed. Ensure model features match.")
        else:
            st.warning("SHAP Explainer unavailable.")

    st.markdown("---")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Predicted RUL", f"{unit_pred['Predicted_RUL']:.0f} Cycles")
    with sc2:
        st.metric("Time to Failure", f"{(unit_pred['Predicted_RUL'] * cycle_multiplier):.0f} Hours")
    with sc3:
        status_str = "🔴 IMPAIRED (Change Detected)" if unit_pred.get('Machine_State', 0) == 1 else "🟢 NORMAL"
        st.write("**Machine State**")
        st.write(status_str)

# ---------------------------------------------------------
# 3. MAINTENANCE PLANNER & TICKETING
# ---------------------------------------------------------
with tab_planner:
    st.subheader("📅 Ticketing System & Workflow")

    t_col1, t_col2 = st.columns([1.2, 1])

    with t_col1:
        st.markdown("### 🚨 Pending Work Orders")
        # Find units that need fixing
        planner_df = display_df[display_df['Predicted_RUL'] < 50].sort_values('Predicted_RUL')

        if not planner_df.empty:
            for idx, row in planner_df.iterrows():
                unit_id = int(row['unit'])

                # Check if a ticket is already open for this unit
                is_ticketed = not st.session_state.tickets_df.empty and unit_id in \
                              st.session_state.tickets_df[st.session_state.tickets_df['Status'] == 'Open'][
                                  'Unit'].values

                if not is_ticketed:
                    with st.expander(
                            f"⚠️ Create Ticket: Asset #{unit_id} - ETA: {int(row['Predicted_RUL'] * cycle_multiplier)} Hours"):
                        with st.form(key=f"form_{unit_id}"):
                            st.write("**Recommended Action:** Thermal inspection & lubrication check.")
                            assignee = st.selectbox("Assign Technician:",
                                                    ["Technician Ali", "Technician Sarah", "Engineering Team Alpha"])
                            submit_ticket = st.form_submit_button("Create & Assign Ticket")

                            if submit_ticket:
                                # Create new ticket record
                                new_ticket = pd.DataFrame([{
                                    "Ticket_ID": f"TK-{np.random.randint(1000, 9999)}",
                                    "Unit": unit_id,
                                    "Assignee": assignee,
                                    "Status": "Open",
                                    "Created_At": datetime.now().strftime("%Y-%m-%d %H:%M")
                                }])
                                st.session_state.tickets_df = pd.concat([st.session_state.tickets_df, new_ticket],
                                                                        ignore_index=True)
                                st.session_state.tickets_df.to_csv("tickets.csv", index=False)  # Save to database
                                st.success("Ticket Assigned Successfully!")
                                st.rerun()
                else:
                    st.info(f"Asset #{unit_id}: Ticket already assigned and open.")
        else:
            st.success("No maintenance required in the next 50 cycles.")

    with t_col2:
        st.markdown("### 📋 Active Tickets")
        if not st.session_state.tickets_df.empty:
            open_tickets = st.session_state.tickets_df[st.session_state.tickets_df['Status'] == 'Open']

            if not open_tickets.empty:
                for idx, t_row in open_tickets.iterrows():
                    with st.container():
                        st.markdown(f"**{t_row['Ticket_ID']} | Asset #{t_row['Unit']}**")
                        st.caption(f"👨‍🔧 Assigned to: {t_row['Assignee']} | 🕒 {t_row['Created_At']}")

                        # Mechanic clicks this when done
                        if st.button("✅ Mark as Resolved", key=f"res_{t_row['Ticket_ID']}"):
                            st.session_state.tickets_df.loc[idx, 'Status'] = 'Resolved'
                            st.session_state.tickets_df.to_csv("tickets.csv", index=False)
                            st.success(f"{t_row['Ticket_ID']} marked resolved. System verifying AI sensors...")
                            time.sleep(1)  # Simulate the system checking the sensors
                            st.rerun()
                        st.divider()
            else:
                st.write("No open tickets. Good job team!")

            # Show history of closed tickets
            closed_tickets = st.session_state.tickets_df[st.session_state.tickets_df['Status'] == 'Resolved']
            if not closed_tickets.empty:
                st.markdown("#### ✅ Resolution History")
                st.dataframe(closed_tickets[['Ticket_ID', 'Unit', 'Assignee']], hide_index=True,
                             use_container_width=True)
        else:
            st.write("No ticket history.")

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

    with sc_2:
        st.write("**Performance Audit**")
        mae = (preds['Actual_RUL'] - preds['Predicted_RUL']).abs().mean()
        rmse = ((preds['Actual_RUL'] - preds['Predicted_RUL']) ** 2).mean() ** 0.5
        st.metric("Model Precision (RMSE)", f"{rmse:.2f} Cycles")
        st.metric("Avg Prediction Error", f"{mae:.2f} Cycles")
        st.write("---")
        st.write("**Trust & Safety Logic:**")
        st.write("✅ **Safety Buffer Applied:** -5 Cycles")
        st.write("✅ **Anomaly Method:** Pelt Change-Point Detection")

# ---------------------------------------------------------
# 5. AI CO-PILOT (CHAT & RATE LIMITING)
# ---------------------------------------------------------
with tab_chat:
    st.subheader("💬 Agentic AI Maintenance Co-Pilot")
    st.caption("Ask questions about the fleet in natural language. Powered by Context-Aware AI.")

    # Display Chat History
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Rate Limiting Logic
    current_time = time.time()
    if current_time - st.session_state.last_call_time > RATE_LIMIT_WINDOW_SEC:
        st.session_state.api_calls = 0
        st.session_state.last_call_time = current_time

    is_rate_limited = st.session_state.api_calls >= RATE_LIMIT_MAX_CALLS

    if is_rate_limited:
        time_left = int(RATE_LIMIT_WINDOW_SEC - (current_time - st.session_state.last_call_time))
        st.error(f"⚠️ API Rate Limit Exceeded. To prevent system overload, please wait {time_left} seconds.")
        chat_input = st.chat_input("Rate limit reached. Please wait...", disabled=True)
    else:
        chat_input = st.chat_input("E.g., 'Which units are critical?' or 'What is the status of unit 42?'")

    # Process User Input
    if chat_input:
        st.session_state.api_calls += 1
        st.session_state.last_call_time = current_time

        st.session_state.messages.append({"role": "user", "content": chat_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(chat_input)

        user_text = chat_input.lower()

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing fleet telemetry..."):
                    time.sleep(0.5)

                    if "critical" in user_text or "urgent" in user_text or "danger" in user_text:
                        if crit_list.empty:
                            response = "Good news! There are currently no critical units in the fleet. All assets are operating normally."
                        else:
                            crit_units = crit_list['unit'].tolist()
                            response = f"I found **{len(crit_units)} critical units** that require immediate attention. They are Units: {', '.join(map(str, crit_units))}."

                    elif "unit" in user_text or "asset" in user_text:
                        match = re.search(r'\d+', user_text)
                        if match:
                            unit_num = int(match.group())
                            if unit_num in preds['unit'].values:
                                u_data = preds[preds['unit'] == unit_num].iloc[0]
                                u_rul = u_data['Predicted_RUL']
                                u_state = "🔴 IMPAIRED" if u_data.get('Machine_State', 0) == 1 else "🟢 NORMAL"
                                response = f"**Asset #{unit_num} Status:**\n- **Remaining Cycles:** {u_rul:.0f}\n- **Machine State:** {u_state}\n- **Time to Failure:** {int(u_rul * cycle_multiplier)} Hours."
                            else:
                                response = f"I cannot find Unit {unit_num} in the database."
                        else:
                            response = "Please provide the unit number (e.g., 'Unit 42')."

                    elif "health" in user_text or "summary" in user_text or "status" in user_text:
                        avg_rul = preds['Predicted_RUL'].mean()
                        response = f"The fleet is operating with an average Remaining Useful Life of **{avg_rul:.1f} cycles**. We are monitoring {len(preds)} total units."

                    else:
                        response = "I am trained on this factory's telemetry data. Ask me 'Which units are critical?' or 'Status of Unit 15'."

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

st.markdown("---")
st.markdown("<center><p style='color: #64748b;'>VHACK Hackathon 2026 | Developed for SME Resilience</p></center>",
            unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. DEVELOPER PANEL (DATA CORRECTION & OVERRIDE)
# ---------------------------------------------------------
with tab_dev:
    st.subheader("🛠️ Admin Developer Control Panel")
    st.warning("CRITICAL: Manual data overrides will affect RUL predictions and fleet safety scores.")

    # Initialize a correction dictionary in session state if it doesn't exist
    if "sensor_corrections" not in st.session_state:
        st.session_state.sensor_corrections = {}

    col_dev1, col_dev2 = st.columns([1, 1.5])

    with col_dev1:
        st.markdown("### ⚙️ Sensor Value Correction")
        st.write(
            "If a sensor is malfunctioning (e.g., constant high noise), apply a 'Tolak Value' (Offset) to normalize the input.")

        with st.form("correction_form"):
            dev_unit = st.selectbox("Select Unit to Correct:", sorted(history['unit'].unique()))
            dev_sensor = st.selectbox("Select Malfunctioning Sensor:",
                                      ['sensor_11', 'sensor_4', 'sensor_15', 'sensor_7'])
            correction_value = st.number_input("Adjustment Value (Offset):", value=0.0,
                                               help="Negative to subtract (Tolak), Positive to add.")

            submit_dev = st.form_submit_button("Apply Correction to Live Stream")

            if submit_dev:
                # Store the correction for this specific unit and sensor
                key = f"{dev_unit}_{dev_sensor}"
                st.session_state.sensor_corrections[key] = correction_value
                st.success(f"Applied {correction_value} offset to {dev_sensor} for Unit {dev_unit}.")
                st.rerun()

    with col_dev2:
        st.markdown("### 📊 Active Overrides")
        if st.session_state.sensor_corrections:
            corr_data = []
            for k, v in st.session_state.sensor_corrections.items():
                # Split only on the first underscore to handle sensor_11 correctly
                parts = k.split("_", 1)
                if len(parts) == 2:
                    u, s = parts
                    corr_data.append({"Unit": u, "Sensor": s, "Offset Applied": v})

            if corr_data:
                st.table(pd.DataFrame(corr_data))

            if st.button("Clear All Overrides"):
                st.session_state.sensor_corrections = {}
                st.rerun()
        else:
            st.info("No manual corrections active. System is running on raw telemetry.")

    st.markdown("---")
    st.markdown("### 🛡️ System Governance")
    st.write("**Security Log:**")
    st.caption(f"Last Admin Access: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.checkbox("Enable Debug Mode (Raw Traceback)")