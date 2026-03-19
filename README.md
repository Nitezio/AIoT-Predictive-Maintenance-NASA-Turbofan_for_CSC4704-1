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

### 1. Advanced Feature Engineering (The SME Resilience Engine)
Industrial IoT data is inherently noisy. To handle this, we implemented:
*   **Temporal Momentum (EMA):** Uses Exponential Moving Averages to capture machine momentum, reacting faster to sudden sensor spikes than standard rolling means.
*   **Multi-Scale Volatility:** Tracks standard deviation and ranges across multiple time windows (10 and 30 cycles) to identify increasing vibration or heat fluctuations.
*   **Robust Imputation:** Uses a time-series-aware Forward-Fill/Backward-Fill strategy to maintain context even if a sensor temporarily drops out.

### 2. High-Dependability ML Core
The system utilizes a **HistGradientBoostingRegressor** with **Sample Weighting**. 
*   **The Logic:** We penalize errors more heavily as the machine approaches end-of-life. 
*   **The Result:** The model is mathematically "forced" to be most accurate during the critical failure phase (0-30 cycles left).

### 3. Anomaly Change-Point Detection (The Fail-Safe)
Independent of the countdown timer, we use the **Pelt Algorithm** (via the `ruptures` library) to monitor the "state" of the machine.
*   It identifies the exact moment a machine's behavior shifts from "Healthy" to "Impaired."
*   If this shift occurs, the system triggers a **State-Change Alert**, providing a secondary layer of protection regardless of the predicted RUL number.

---

## 🛡️ Risk Management: Mitigating "Dangerous Errors"

In predictive maintenance, a "Dangerous Error" occurs when the AI is too optimistic. Our system uses three layers of defense to ensure these errors never lead to disaster:

1.  **Dynamic Update Convergence:** Predictions are not static. Because the AI re-evaluates the machine every cycle, an optimistic error today is corrected tomorrow as the sensor data becomes more "violent" near failure. 
2.  **State-Override Alert:** Even if the RUL countdown says "50 cycles left," the Anomaly Detection module (Brain B) will flip the machine to **"🔴 IMPAIRED"** the moment a change-point is detected, overriding any optimistic numbers.
3.  **Human-in-the-Loop (XAI):** By providing SHAP waterfall plots, we allow human technicians to see *why* the AI is predicting health. If sensors are spiking but the number is high, the technician can manually intervene using the **Developer Control Room**.

---

## 📊 Performance & Reliability Audit

The system has been subjected to rigorous multi-phase testing against the NASA C-MAPSS dataset.

### 1. Algorithmic Stability Benchmark (5-Trial Summary)
| Trial | RMSE | MAE | R² Score | System F1 | Dangerous Errors |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | 18.50 | 13.04 | 0.8018 | 0.9111 | 13 |
| 2 | 18.41 | 13.07 | 0.8037 | 0.9319 | 14 |
| 3 | 18.25 | 13.02 | 0.8070 | 0.9200 | 13 |
| 4 | 18.22 | 12.72 | 0.8077 | 0.9309 | 12 |
| 5 | 18.66 | 13.16 | 0.7983 | 0.9200 | 14 |
| **AVERAGE** | **18.41** | **13.00** | **0.8037** | **0.9228** | **13.2** |

### 2. Consolidated Reliability Report (Production Model)
| Category | Theoretical Max | Safe Production | Status |
| :--- | :---: | :---: | :--- |
| **Statistical RMSE** | **18.41** | 19.21 | ✅ Optimized |
| **System F1-Score** | **0.9228** | **0.9123** | 🚀 Elite |
| **Dangerous Errors** | 13.2 Units | **8 Units** | 🛡️ **Ultra-Safe** |
| **Critical Catch (Recall)**| 85% | **92%** | 🛠️ High Reliability |

---

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the AI Engine
This script trains the model and generates the necessary analytical artifacts.
```bash
python train_model.py
```

### 3. Launch the AIoT Command Center
```bash
streamlit run dashboard.py
```

---
🛡️ **Developed for Industrial Resilience & Sustainable Infrastructure (SDG 9)**
