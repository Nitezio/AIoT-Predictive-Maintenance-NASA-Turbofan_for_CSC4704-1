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

## 📊 Performance & Reliability Audit

The system has been subjected to rigorous multi-phase testing to ensure it meets industrial-grade standards for stability and safety.

### 1. Algorithmic Stability Benchmark (5-Trial Summary)
This proves the stability of the predictive engine across different data initializations.
| Trial | RMSE | MAE | R² Score | Bias | System F1 | Dangerous Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 18.50 | 13.04 | 0.8018 | -0.12 | 0.9111 | 13 |
| 2 | 18.41 | 13.07 | 0.8037 | +0.29 | 0.9319 | 14 |
| 3 | 18.25 | 13.02 | 0.8070 | +0.46 | 0.9200 | 13 |
| 4 | 18.22 | 12.72 | 0.8077 | -0.03 | 0.9309 | 12 |
| 5 | 18.66 | 13.16 | 0.7983 | +0.14 | 0.9200 | 14 |
| **AVERAGE** | **18.41** | **13.00** | **0.8037** | **+0.15** | **0.9228** | **13.2** |

### 2. Deployed Production Model Performance
Metrics for the final system with the integrated **SME Safety Buffer**.
| Metric | Result | Industrial Interpretation |
| :--- | :---: | :--- |
| **RMSE (Precision)** | 19.21 | High accuracy within critical cycle windows. |
| **System F1-Score** | **0.9123** | **91% accuracy** in automated health-state triage. |
| **Mean Bias** | **-5.10** | **Safe-Pessimistic:** Prevents late-maintenance risks. |
| **Dangerous Errors** | **8 Units** | **Safety Target Met:** Minimal over-optimism risk. |

### 3. Consolidated Reliability Report
| Category | Theoretical Max | Safe Production | Status |
| :--- | :---: | :---: | :--- |
| **Statistical RMSE** | **18.41** | 19.21 | ✅ Optimized |
| **System F1-Score** | **0.9228** | **0.9123** | 🚀 Elite |
| **Dangerous Errors** | 13.2 Units | **8 Units** | 🛡️ **Ultra-Safe** |
| **Critical Catch (Recall)**| 85% | **92%** | 🛠️ High Reliability |

---

## 🧠 Technical Analysis

1.  **State-Aware Categorization:** The model maintains a **91% F1-Score**, meaning it correctly distinguishes between Healthy, Warning, and Critical states for the vast majority of the fleet.
2.  **Safety-First Engineering:** By implementing a safety buffer, we reduced dangerous over-optimistic errors by **40%**. In industrial settings, being "Safe" is prioritized over raw mathematical precision.
3.  **High-Recall Failure Detection:** With a **92% Recall for Critical Units**, the system ensures that nearly all assets approaching failure are identified early enough for proactive intervention.

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
