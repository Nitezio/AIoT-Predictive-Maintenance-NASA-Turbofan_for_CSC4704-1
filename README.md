# AIoT Predictive Maintenance - NASA Turbofan Engine

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue.svg)
![License](https://img.shields.io/badge/License-Academic%20Project-lightgrey.svg)

A machine learning project for predictive maintenance using NASA's Turbofan Engine Degradation dataset. This academic project implements LSTM neural networks to predict Remaining Useful Life (RUL) of turbofan engines and provides an interactive dashboard for visualization.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Model Architecture](#model-architecture)
- [Dashboard Features](#dashboard-features)
- [Results](#results)
- [Academic Context](#academic-context)
- [Contributors](#contributors)
- [License](#license)

---

## 🎯 Project Overview

This project explores predictive maintenance techniques using deep learning on the NASA Turbofan Engine Degradation Simulation Dataset. The system predicts the Remaining Useful Life (RUL) of aircraft engines based on sensor readings and operational settings.

**Key Features:**
- LSTM-based deep learning model for RUL prediction
- Interactive Streamlit dashboard for data visualization
- Comprehensive data preprocessing pipeline
- Model performance analysis and visualization
- CSV-based data loading for dashboard

**⚠️ IMPORTANT: This is an academic project. Model training must be completed before using the dashboard.**

---

## 📊 Dataset Information

**Source:** NASA Prognostics Data Repository - Turbofan Engine Degradation Simulation Data Set

**Dataset Characteristics:**
- Multiple multivariate time series from engine run-to-failure
- 21 sensor measurements per engine cycle
- 3 operational settings
- Engine degradation over time until failure
- Training and test subsets for model validation

**Data Files:**
- `train_FD001.txt` - Training data
- `test_FD001.txt` - Test data  
- `RUL_FD001.txt` - Ground truth RUL values for test set

---

## 📁 Project Structure

```
AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1/
│
├── data/                          # Dataset directory
│   ├── train_FD001.txt           # Training dataset
│   ├── test_FD001.txt            # Test dataset
│   └── RUL_FD001.txt             # Ground truth RUL values
│
├── models/                        # Saved model files
│   ├── lstm_model.h5             # Trained LSTM model (generated)
│   └── scaler.pkl                # Data scaler (generated)
│
├── output/                        # Model outputs and results
│   ├── predictions.csv           # Model predictions (generated)
│   ├── training_history.png      # Training curves (generated)
│   └── prediction_plot.png       # Prediction visualization (generated)
│
├── train_model.py                # Main training script
├── dashboard.py                  # Streamlit dashboard application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation

```

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural network API

### Data Processing & Analysis
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Data preprocessing and metrics

### Visualization & Dashboard
- **Streamlit** - Interactive web dashboard
- **Matplotlib** - Static plotting and visualizations
- **Seaborn** - Statistical data visualization

### Model Architecture
- **LSTM Networks** - Sequential data modeling
- **Dense Layers** - Fully connected neural networks
- **Dropout** - Regularization technique

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step-by-Step Installation

1. **Clone the Repository**
```bash
git clone https://github.com/Nitezio/AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1.git
cd AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1
```

2. **Create Virtual Environment (Recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Installation**
```bash
python -c "import tensorflow as tf; import streamlit as st; print('Installation successful!')"
```

---

## 📖 Usage Guide

### ⚠️ CRITICAL: Training Must Be Completed First

**The dashboard requires trained model outputs to function. You MUST run the training script before using the dashboard.**

### Step 1: Train the Model (REQUIRED)

```bash
python train_model.py
```

**What this script does:**
1. Loads and preprocesses the NASA Turbofan dataset from `data/` directory
2. Engineers features and prepares sequences for LSTM input
3. Builds and trains the LSTM neural network
4. Evaluates model performance on test data
5. **Generates required files:**
   - `models/lstm_model.h5` - Trained model
   - `models/scaler.pkl` - Data scaler
   - `output/predictions.csv` - Predictions for dashboard
   - `output/training_history.png` - Training visualization
   - `output/prediction_plot.png` - Results visualization

**Expected Output:**
```
Loading data...
Preprocessing data...
Building LSTM model...
Training model...
Epoch 1/50 - loss: 0.0234 - val_loss: 0.0198
...
Model saved to models/lstm_model.h5
Predictions saved to output/predictions.csv
Training completed successfully!
```

**Training Time:** Approximately 10-30 minutes depending on hardware

### Step 2: Launch the Dashboard

```bash
streamlit run dashboard.py
```

**What the dashboard does:**
- Loads predictions from `output/predictions.csv`
- Displays interactive visualizations
- Shows model performance metrics
- Provides data exploration tools

**Dashboard Access:**
- Local URL: `http://localhost:8501`
- Network URL: Will be displayed in terminal

---

## 🧠 Model Architecture

### LSTM Network Configuration

```python
Model: Sequential LSTM Network

Layer 1: LSTM(100 units, return_sequences=True)
         - Input: 3D tensor (samples, timesteps, features)
         - Output: 3D tensor (samples, timesteps, 100)
         
Layer 2: Dropout(0.2)
         - Regularization to prevent overfitting
         
Layer 3: LSTM(50 units, return_sequences=False)
         - Output: 2D tensor (samples, 50)
         
Layer 4: Dropout(0.2)
         - Additional regularization
         
Layer 5: Dense(1 unit, activation='linear')
         - Output: Predicted RUL value

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Metrics: Mean Absolute Error (MAE)
```

### Data Preprocessing Pipeline

1. **Feature Selection:** 21 sensor readings + 3 operational settings
2. **Normalization:** Min-Max scaling to [0, 1] range
3. **Sequence Creation:** Time windows for LSTM input
4. **RUL Calculation:** Cycles remaining until failure
5. **Train-Test Split:** Separate datasets for validation

---

## 📊 Dashboard Features

### Dashboard Overview

The Streamlit dashboard provides comprehensive visualization and analysis tools. **Note: The dashboard loads data from CSV files generated during training, not from a live database.**

### Main Components

#### 1. **Overview Section**
- Project summary and objectives
- Dataset statistics
- Model architecture visualization

#### 2. **Data Exploration**
- Sensor readings visualization
- Engine degradation patterns
- Statistical summaries
- Time series plots

#### 3. **Model Performance**
- Training history (loss curves)
- Prediction accuracy metrics
- Error distribution analysis
- Residual plots

#### 4. **Prediction Analysis**
- Actual vs Predicted RUL comparison
- Individual engine predictions
- Confidence intervals
- Error analysis by engine

#### 5. **Interactive Visualizations**
- Plotly-based interactive charts
- Zoom, pan, and hover capabilities
- Customizable plot parameters
- Export functionality

### Data Source Clarification

**The dashboard operates on pre-generated CSV files:**
- `output/predictions.csv` - Contains all prediction data
- No real-time database connection
- Data refreshes when training is re-run
- Suitable for academic demonstration purposes

---

## 📈 Results

### Model Performance Metrics

**Training Performance:**
- Training Loss (MSE): ~0.015
- Validation Loss (MSE): ~0.018
- Training MAE: ~8.5 cycles
- Validation MAE: ~9.2 cycles

**Test Set Results:**
- Test MSE: ~0.020
- Test MAE: ~10.1 cycles
- R² Score: ~0.85

**Note:** Actual results may vary based on random initialization and training conditions.

### Key Findings

1. **Model Accuracy:** LSTM architecture successfully captures temporal degradation patterns
2. **Prediction Horizon:** Best accuracy within 50 cycles of failure
3. **Feature Importance:** Specific sensors show stronger correlation with RUL
4. **Generalization:** Model performs consistently across different engines

---

## 🎓 Academic Context

**Course:** CSC4704-1  
**Project Type:** Academic Research Project  
**Purpose:** Educational exploration of predictive maintenance using AI/ML

### Learning Objectives Met

- ✅ Implementation of LSTM neural networks
- ✅ Time series data analysis and preprocessing
- ✅ Deep learning model training and evaluation
- ✅ Interactive dashboard development
- ✅ End-to-end ML pipeline creation

### Educational Value

This project demonstrates:
- Real-world application of deep learning
- Handling of multivariate time series data
- Model evaluation and validation techniques
- Software engineering for ML projects
- Data visualization best practices

**Disclaimer:** This is an academic project for learning purposes. It is not intended for production use or commercial deployment.

---

## 👥 Contributors

- **Project Team:** CSC4704-1 Students
- **Repository Owner:** Nitezio

### Contributions Welcome

This is an academic project, but suggestions and improvements are welcome:
- Bug reports
- Documentation improvements
- Code optimization suggestions
- Feature enhancement ideas

---

## 📄 License

This project is developed for academic purposes as part of CSC4704-1 coursework.

**Dataset License:** NASA Prognostics Data Repository (Public Domain)

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: Dashboard shows "File not found" error**
```
Solution: Run train_model.py first to generate required output files
```

**Issue 2: Import errors for TensorFlow/Keras**
```
Solution: Ensure you're using Python 3.8+ and reinstall dependencies:
pip install --upgrade -r requirements.txt
```

**Issue 3: Memory errors during training**
```
Solution: Reduce batch size in train_model.py or use a machine with more RAM
```

**Issue 4: Streamlit dashboard won't start**
```
Solution: Check if port 8501 is available or specify different port:
streamlit run dashboard.py --server.port 8502
```

---

## 📚 References

1. NASA Prognostics Data Repository - Turbofan Engine Degradation Simulation
2. Hochreiter & Schmidhuber (1997) - Long Short-Term Memory Networks
3. TensorFlow/Keras Documentation
4. Streamlit Documentation

---

## 🚀 Future Enhancements (Academic Ideas)

- [ ] Implement multiple LSTM architectures for comparison
- [ ] Add ensemble methods for improved accuracy
- [ ] Integrate additional NASA datasets (FD002, FD003, FD004)
- [ ] Develop real-time prediction capability
- [ ] Add model explainability features (SHAP, LIME)
- [ ] Implement hyperparameter optimization
- [ ] Create automated retraining pipeline

---

## 📞 Contact & Support

For academic inquiries or project-related questions:
- **GitHub Issues:** [Report issues here](https://github.com/Nitezio/AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1/issues)
- **Repository Owner:** Nitezio

---

**Last Updated:** 2026-01-02

**Project Status:** Active (Academic Project)

---

*This project is part of CSC4704-1 coursework and serves as an educational demonstration of AI/IoT applications in predictive maintenance.*
