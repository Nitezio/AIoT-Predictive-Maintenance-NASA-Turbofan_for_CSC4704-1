# 🛡️ Resilient AIoT Predictive Maintenance for Industrial SMEs

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![SDG 9](https://img.shields.io/badge/SDG-9-orange.svg)](https://sdgs.un.org/goals/goal9)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

## 📖 Executive Summary

Small and Medium Enterprises (SMEs) are the backbone of the global economy, yet they often struggle with aging machinery and reactive maintenance cultures. A single motor failure in a production plant can halt operations for weeks, leading to significant financial loss.

This **AIoT Command Center** transitions predictive maintenance from a luxury for large conglomerates into a robust, resilient tool for local SMEs. By integrating advanced machine learning with **Anomaly Change-Point Detection**, **Explainable AI (XAI)**, and **Agentic AI Assistance**, this system provides factory managers with clear, actionable insights to prevent downtime and optimize resource usage.

---

## 🌟 Key Features & Innovations

*   **🤖 Agentic AI Co-Pilot:** A natural language interface that allows managers to "chat" with their machinery fleet for instant health updates and unit-specific status.
*   **🎫 Industrial Ticketing Workflow:** A built-in digital work order system to assign technical staff and track repair resolutions in real-time.
*   **🧠 Explainable AI (SHAP):** Local interpretability module that explains *why* the AI predicts a specific failure, building critical trust with equipment operators.
*   **🛠️ Developer Control Room:** Governance feature allowing manual sensor offsets to handle hardware drift or malfunctioning IoT devices without system downtime.
*   **⚠️ Fail-Safe Safety Buffer:** A built-in pessimistic bias that ensures maintenance alerts are triggered with a safety cushion before catastrophic failures occur.

---

## 🔬 Methodology & System Architecture

Our system employs a **Defense-in-Depth** architecture, combining three independent AI methods to ensure no failure goes undetected.

### 1. Advanced Feature Engineering
Industrial IoT data is inherently noisy. To handle this, we implemented:
*   **Temporal Momentum (EMA):** Captures machine momentum, reacting faster to sudden sensor spikes.
*   **Multi-Scale Volatility:** Tracks fluctuations across 10 and 30 cycle windows to identify increasing vibrations.
*   **Robust Imputation:** Time-series-aware strategy to maintain context during sensor dropouts.

### 2. High-Dependability ML Core
Utilizes a **HistGradientBoostingRegressor** with **Sample Weighting** to prioritize accuracy during the critical failure phase (0-30 cycles left).

### 3. Anomaly Change-Point Detection
Uses the **Pelt Algorithm** to monitor machine behavior independently of the countdown, identifying the exact moment a machine transitions to an "Impaired" state.

---

## 📊 Performance & Reliability Audit

The following results are derived from the latest rigorous testing against the NASA C-MAPSS industrial dataset.

### 1. Algorithmic Stability Benchmark (Raw AI)
*Measures raw mathematical intelligence across 5 trials with different data initializations.*
| Trial | RMSE | MAE | R² Score | Bias | System F1 | Dangerous Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 18.50 | 13.04 | 0.8018 | -0.12 | 0.9111 | 13 |
| 2 | 18.41 | 13.07 | 0.8037 | +0.29 | 0.9319 | 14 |
| 3 | 18.25 | 13.02 | 0.8070 | +0.46 | 0.9200 | 13 |
| 4 | 18.22 | 12.72 | 0.8077 | -0.03 | 0.9309 | 12 |
| 5 | 18.66 | 13.16 | 0.7983 | +0.14 | 0.9200 | 14 |
| **AVG** | **18.41** | **13.00** | **0.8037** | **+0.15** | **0.9228** | **13.2** |

### 2. Safe Production Stability Benchmark (With Buffer)
*Measures the stability of the final deployed system with the -5 cycle safety buffer enabled.*
| Trial | RMSE | MAE | Bias | System F1 | Dangerous Errors |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | 19.21 | 13.94 | -5.10 | 0.9123 | 8 |
| 2 | 19.01 | 13.65 | -4.68 | 0.9123 | 8 |
| 3 | 18.83 | 13.47 | -4.49 | 0.9034 | 8 |
| 4 | 18.92 | 13.62 | -4.98 | 0.9109 | 7 |
| 5 | 19.31 | 13.96 | -4.83 | 0.9223 | 7 |
| **AVG** | **19.06** | **13.73** | **-4.82** | **0.9122** | **7.6** |

### 3. Consolidated Reliability Report
| Category | Theoretical Max (Raw) | Safe Production (Buffer) | Status |
| :--- | :---: | :---: | :--- |
| **Statistical RMSE** | 18.41 | 19.06 | ✅ Balanced |
| **System F1-Score** | 0.9228 | **0.9122** | 🚀 Elite Triage |
| **Mean Bias** | +0.15 (Neutral) | **-4.82 (Pessimistic)** | 🛡️ **Fail-Safe** |
| **Dangerous Errors** | 13.2 Units | **7.6 Units** | ✅ Target Met (<10) |
| **Critical Recall** | 85% | **92%** | 🛠️ High Reliability |

---

## 🧠 Final Analysis of Results

*   **Reliability through Consistency:** The average **System F1-Score of 0.9122** proves that the AI correctly triages 91% of the fleet into the correct health categories (Healthy vs. Warning vs. Critical).
*   **The Safety Advantage:** By moving to the Production version, we successfully **reduced Dangerous Errors by 42%** (13.2 down to 7.6). This shift from a neutral bias (+0.15) to a pessimistic bias (-4.82) ensures that factory managers are warned before failures occur.
*   **High-Stakes Precision:** With a **92% Recall for Critical Assets**, the system is highly dependable for catching machines nearing end-of-life, providing SMEs with the resilience to maintain continuous operations.

---

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the AI Engine
```bash
python train_model.py
```

### 3. Launch the AIoT Command Center
```bash
streamlit run dashboard.py
```

---
🛡️ **Developed for Industrial Resilience & Sustainable Infrastructure (SDG 9)**
