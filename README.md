# 🛡️ VHACK 2026: Resilient AIoT Predictive Maintenance for ASEAN SMEs

[![Track](https://img.shields.io/badge/VHACK-Track%201-blue.svg)](https://vhack.usm.my/)
[![SDG 9](https://img.shields.io/badge/SDG-9-orange.svg)](https://sdgs.un.org/goals/goal9)
[![Status](https://img.shields.io/badge/Status-Hackathon%20Upgrade-success.svg)]()

### 🏆 Varsity Hackathon 2026 Submission
**Case Study 1:** Predictive Maintenance for SME Resilience  
**Primary Goal:** SDG 9: Industry, Innovation, and Infrastructure (Target 9.4)

---

## 📖 Executive Summary

Small and Medium Enterprises (SMEs) are the backbone of the ASEAN economy, yet they often struggle with aging machinery and reactive maintenance cultures. A single motor failure in a rural food processing plant can halt production for weeks. 

Our **VHACK AIoT Command Center** transitions predictive maintenance from a luxury for conglomerates into a robust, resilient tool for local SMEs. By using advanced ML with **Anomaly Change-Point Detection** and **Explainable AI (XAI)**, we provide factory managers with clear, actionable insights to prevent downtime and optimize resource usage.

### 🌟 Key Hackathon Enhancements

*   **🔍 Anomaly Change-Point Detection:** Implemented `ruptures` (Pelt algorithm) to identify the exact moment a machine transitions from "Healthy" to "Impaired", allowing for earlier intervention.
*   **🧠 Explainable AI (SHAP):** Local interpretability module that explains *why* the AI predicts a failure, building trust with non-technical operators.
*   **🛡️ SME Resilience Engine:** Robust data imputation (Forward-fill/Backward-fill) to handle noisy or missing sensor data common in industrial settings.
*   **🛠️ Proactive Maintenance Scheduler:** Automated work-order generation and "Time-to-Failure" conversions (Cycles to Hours) for real-world usability.
*   **🌍 Multi-lingual Accessibility:** Localized alerts (e.g., Malay) to ensure inclusivity in local factory environments.

---

## 📂 Architecture

```text
Group_Project_Topic4/
│
├── dashboard.py               # 📊 VHACK Command Center (Streamlit + SHAP + Scheduler)
├── train_model.py             # 🧠 AI Engine (Change-Point + XAI Explainer + Robust Imputation)
├── requirements.txt           # 📦 Updated dependencies (shap, ruptures)
├── README.md                  # 📄 Hackathon Pitch & Documentation
│
├── data/                      # 🗄️ NASA C-MAPSS FD001 Dataset
│   ├── train_FD001.txt        
│   ├── test_FD001.txt         
│   └── RUL_FD001.txt          
│
└── artifacts/                 # Generated locally (Excluded from Git)
    ├── model.pkl
    ├── shap_explainer.pkl     # XAI Explainer
    ├── processed_test_history.csv
    └── predictions.csv
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the AI Engine (Mandatory)
This script trains the model, detects change-points, and generates the XAI explainer.
```bash
python train_model.py
```

### 3. Launch the VHACK Command Center
```bash
streamlit run dashboard.py
```

---

## 🛠️ Technical Feasibility & Constraints

### Noisy Data Handling
Unlike standard models that drop missing values, our pipeline uses a time-series-aware imputation strategy. This ensures that even if an IoT sensor drops out for 5 cycles, the model retains the degradation context.

### Model Interpretability (XAI)
Using **SHAP (SHapley Additive exPlanations)**, the dashboard provides a waterfall plot for every engine. If the RUL is low, the AI explicitly states: *"Sensor 11 volatility is the primary driver for this maintenance alert."*

### Scalability
The core model is a Random Forest Regressor, chosen for its lightweight nature. It can be deployed on edge devices like Raspberry Pi or low-cost industrial gateways, making it ideal for budget-constrained SMEs.

---

## 🌍 Social Impact (SDG 9.4)

By 2030, upgrade infrastructure and retrofit industries to make them sustainable. Our solution:
1.  **Reduces Waste:** Prevents "over-maintenance" (replacing parts too early).
2.  **Prevents Bankruptcy:** Minimizes catastrophic downtime for thin-margin SMEs.
3.  **Local Empowerment:** Multi-lingual alerts bridge the gap between advanced AI and local workforce skills.

---

## 👥 Authors
- **Nitezio** - *Project Lead*
- **Gemini CLI** - *Lead Architect & Implementation*

---
⭐ **Submission for USM Varsity Hackathon 2026**
