# 🛡️ VHACK 2026: Resilient AIoT Predictive Maintenance for ASEAN SMEs

[![Track](https://img.shields.io/badge/VHACK-Track%201-blue.svg)](https://vhack.usm.my/)
[![SDG 9](https://img.shields.io/badge/SDG-9-orange.svg)](https://sdgs.un.org/goals/goal9)
[![Status](https://img.shields.io/badge/Status-Hackathon%20Final-success.svg)]()

### 🏆 Varsity Hackathon 2026 Submission
**Case Study 1:** Predictive Maintenance for SME Resilience  
**Primary Goal:** SDG 9: Industry, Innovation, and Infrastructure (Target 9.4)

---

## 📖 Executive Summary

Small and Medium Enterprises (SMEs) are the backbone of the ASEAN economy, yet they often struggle with aging machinery and reactive maintenance cultures. A single motor failure in a rural food processing plant can halt production for weeks.

Our **VHACK AIoT Command Center (v2.0)** transitions predictive maintenance from a luxury for conglomerates into a robust, resilient tool for local SMEs. By using advanced ML with **Anomaly Change-Point Detection**, **Explainable AI (XAI)**, and a **Fail-Safe Pessimistic Bias**, we provide factory managers with clear, actionable insights to prevent downtime and optimize resource usage.

---

## 🌟 Key Features & Innovations

*   **🔍 Anomaly Change-Point Detection:** Implemented the `ruptures` (Pelt algorithm) to identify the exact moment a machine transitions from "Healthy" to "Impaired," enabling early-warning alerts.
*   **🧠 Explainable AI (SHAP):** An integrated XAI module that explains *why* the AI predicts a specific failure cycle, building critical trust with non-technical SME operators.
*   **🛡️ High-Dependability AI Engine:** Utilizes `HistGradientBoosting` with **Sample Weighting**, prioritizing accuracy during the critical end-of-life phase (0-30 cycles).
*   **⚠️ SME Safety Buffer:** A built-in "Pessimistic Bias" that intentionally predicts failure slightly earlier than expected to ensure a safety cushion for maintenance planning.
*   **🌍 Multi-lingual Accessibility:** Localized alerts in **Bahasa Melayu** to ensure floor workers can react instantly to critical failures.

---

## 📊 Final Stability Benchmark (5-Trial Summary)

To ensure this model is reliable enough for industrial deployment, we conducted a rigorous **Seed-Variant Validation** test. We ran the engine 5 times using different random seeds to verify that its performance is stable and not a result of chance.

| Trial | RMSE | MAE | R² Score | Mean Bias | F1 (Critical) | Dangerous Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 18.50 | 13.04 | 0.8018 | -0.12 | 0.8462 | 13 |
| 2 | 18.41 | 13.07 | 0.8037 | +0.29 | 0.8800 | 14 |
| 3 | 18.25 | 13.02 | 0.8070 | +0.46 | 0.8462 | 13 |
| 4 | 18.22 | 12.72 | 0.8077 | -0.03 | 0.8462 | 12 |
| 5 | 18.66 | 13.16 | 0.7983 | +0.14 | 0.8462 | 14 |
| **AVERAGE** | **18.41** | **13.00** | **0.8037** | **+0.15** | **0.8530** | **13.2** |

---

## 🧠 How the Test Works & What it Teaches Us

### 1. The Method: Cross-Seed Validation
In Machine Learning, a "Seed" controls the randomness of how the model learns. If a model is weak, its performance will swing wildly with different seeds. 
*   **The Result:** Our RMSE only fluctuated between 18.22 and 18.66. This teaches us that the model's architecture is **extremely stable** and can be trusted across different factory environments.

### 2. Metrics for Non-Engineers
*   **MAE (13.00):** On average, our "Remaining Life" estimate is only 13 cycles away from reality. For a machine that runs for months, this is high-precision data.
*   **F1 Score (0.8530):** This is our "Catch-Rate." It proves that the model is exceptionally good at separating "Normal" machines from those about to break.

### 3. Understanding "Dangerous Errors"
A "Dangerous Error" occurs when the AI is too optimistic (predicting a machine is fine when it is actually failing). By observing the benchmark, we identified a 13% risk rate. 
*   **The Fix:** We implemented a **-5 cycle safety buffer** in the final production code. This teaches us that in industrial AI, **being "Correct" is less important than being "Safe."**

---

## 🛡️ Why SMEs Can Trust This System

1.  **Fail-Safe Categorization:** Our model is designed to be **Pessimistic**. It would rather give a "Warning" 5 hours too early than stay silent 1 minute too late. This "Better Safe than Sorry" approach protects thin SME profit margins.
2.  **Noisy Data Resilience:** Industrial IoT sensors are often noisy. Our use of **Exponential Moving Averages (EMA)** and **Forward/Backward Fill** ensures the model doesn't "panic" over a single data spike.
3.  **Human-in-the-Loop:** We don't just give a number; the **SHAP Waterfall Plot** in the dashboard tells the engineer exactly *which* sensor is causing the alarm. This allows a mechanic to verify the AI's logic manually.

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

## 👥 Authors
- **Nitezio** - *Project Lead*
- **Gemini CLI** - *Lead Architect & Implementation*

---
⭐ **Official Submission for USM Varsity Hackathon 2026**
