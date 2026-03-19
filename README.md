# 🛡️ VHACK 2026: Resilient AIoT Predictive Maintenance for ASEAN SMEs

[![Track](https://img.shields.io/badge/VHACK-Track%201-blue.svg)](https://vhack.usm.my/)
[![SDG 9](https://img.shields.io/badge/SDG-9-orange.svg)](https://sdgs.un.org/goals/goal9)
[![Status](https://img.shields.io/badge/Status-Hackathon%20Final%20v2.1-success.svg)]()

### 🏆 Varsity Hackathon 2026 Submission
**Case Study 1:** Predictive Maintenance for SME Resilience  
**Primary Goal:** SDG 9: Industry, Innovation, and Infrastructure (Target 9.4)

---

## 📖 Executive Summary

Small and Medium Enterprises (SMEs) are the backbone of the ASEAN economy, yet they often struggle with aging machinery and reactive maintenance cultures. A single motor failure in a rural food processing plant can halt production for weeks.

Our **VHACK AIoT Command Center (v2.1)** transitions predictive maintenance from a luxury for conglomerates into a robust, resilient tool for local SMEs. By using advanced ML with **Anomaly Change-Point Detection**, **Explainable AI (XAI)**, and **Agentic AI Assistance**, we provide factory managers with clear, actionable insights to prevent downtime and optimize resource usage.

---

## 🌟 Key Features & Innovations (v2.1)

### 🤖 Agentic AI Co-Pilot
A natural language interface that allows managers to "chat" with the fleet. You can ask: *"Which units are critical?"* or *"What is the health of asset 42?"* and get instant, context-aware answers derived from the latest telemetry.

### 🎫 Industrial Ticketing Workflow
A built-in maintenance management system. When the AI detects a critical unit, a manager can instantly generate a **Digital Work Order**, assign a technician (e.g., Ali or Sarah), and track the resolution in real-time.

### 🧠 Explainable AI (SHAP)
The "Asset Analysis" tab answers the most important question for trust: *"Why does the AI think this machine is failing?"* using SHAP waterfall plots to visualize specific sensor drivers.

### 🛠️ Developer Admin Control Room
A governance feature that allows admins to manually apply **Sensor Offsets**. If a physical hardware sensor begins malfunctioning or drifting, the admin can normalize the data without stopping the factory.

### 🌍 Multi-lingual Alerts
Integrated support for **Bahasa Melayu** to ensure floor workers across the region can understand and react to safety warnings.

---

## 📊 Final Stability Benchmark (5-Trial Summary)

We ran the AI Engine 5 times with different random seeds to verify that the **System F1-Score** (our ability to correctly categorize Healthy, Warning, and Critical states) is stable and robust.

| Trial | RMSE | MAE | R² Score | Mean Bias | System F1 | Dangerous Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 18.50 | 13.04 | 0.8018 | -0.12 | 0.8529 | 13 |
| 2 | 18.41 | 13.07 | 0.8037 | +0.29 | 0.8800 | 14 |
| 3 | 18.25 | 13.02 | 0.8070 | +0.46 | 0.8529 | 13 |
| 4 | 18.22 | 12.72 | 0.8077 | -0.03 | 0.8529 | 12 |
| 5 | 18.66 | 13.16 | 0.7983 | +0.14 | 0.8529 | 14 |
| **AVERAGE** | **18.41** | **13.00** | **0.8037** | **+0.15** | **0.8583** | **13.2** |

---

## 🧠 How the Test Works & What it Teaches Us

### 1. The Method: Cross-Seed Validation
In Machine Learning, a "Seed" controls the learn-path. By keeping our RMSE within a narrow band (18.2 - 18.6), we prove the system is **Dependable** regardless of data variation.

### 2. The Multi-Class System F1 Score
Instead of just measuring one number, the **System F1** measures how accurately the AI places assets into three distinct buckets:
*   **🟢 Healthy:** No action needed.
*   **🟡 Warning:** Schedule inspection.
*   **🔴 Critical:** High risk of failure.
A score of **0.85+** means the AI's "triage" logic is accurate enough for autonomous scheduling.

### 3. Safety vs. Precision
Our production model applies a **-5 cycle safety buffer**. While the benchmark shows raw mathematical precision, the final app is **intentionally pessimistic**. It is better to repair a machine 5 hours early than to have it explode 5 minutes late.

---

## 🛡️ Why SMEs Can Trust This System

1.  **Fail-Safe Categorization:** Prioritizes **Recall** (catching every possible failure) over pure accuracy.
2.  **Noisy Data Resilience:** Industrial IoT sensors drift. Our **EMA (Moving Average)** and **Forward-Fill** logic prevent "false panic" while maintaining degradation context.
3.  **Governance & Audit:** Every action—from sensor overrides to ticket resolution—is logged, providing a clear digital trail for safety audits.

---

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the AI Engine (Mandatory)
```bash
python train_model.py
```

### 3. Launch the Command Center
```bash
streamlit run dashboard.py
```

---
⭐ **Official Submission for USM Varsity Hackathon 2026**
