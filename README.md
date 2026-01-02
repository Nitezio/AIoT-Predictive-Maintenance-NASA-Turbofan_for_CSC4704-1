# 🛡️ AIoT Predictive Maintenance System
### CSC4704 Group Project | Topic 4: Predictive Maintenance using Industrial IoT Data
**Dataset:** NASA C-MAPSS FD001 | **Model:** Random Forest Regressor

---

## 📖 Project Overview

This project implements an end-to-end **Predictive Maintenance Pipeline** for commercial turbofan engines.  By analyzing sensor data from the **NASA C-MAPSS FD001 dataset**, the system predicts the **Remaining Useful Life (RUL)** of engines. This allows maintenance teams to schedule repairs proactively, preventing catastrophic failures and optimizing operational costs.

### Key Features

* **🤖 Machine Learning:** Random Forest Regressor trained on 100 run-to-failure engine trajectories
* **⚙️ Feature Engineering:** Rolling statistics (mean, std) to capture temporal degradation patterns
* **📊 Interactive Dashboard:** Streamlit-based web interface for real-time visualization of fleet health
* **🛠️ Maintenance Insights:** Automated "Critical/Warning/Healthy" status classification with business value estimation

---

## 📂 Directory Structure

This repository is organized to separate source code, data, and generated artifacts.

```text
AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1/
│
├── dashboard.py                # 📊 Interactive Web Dashboard (Streamlit)
├── train_model.py              # 🧠 Main AI Pipeline (Data Processing -> Training -> Evaluation)
├── requirements.txt            # 📦 List of Python dependencies
├── README.md                   # 📄 Project Documentation (You are here)
│
└── data/                       # 🗄️ Raw Dataset Directory
    ├── train_FD001.txt         # Training Data (Run-to-failure)
    ├── test_FD001.txt          # Test Data (Truncated history)
    └── RUL_FD001.txt           # Ground Truth RUL for Test Data
```

---

## 🚀 Installation & Setup

Follow these steps to set up the environment and run the system. 

### 1. Clone the Repository

```bash
git clone https://github.com/Nitezio/AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1.git
cd AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1
```

### 2. Install Dependencies

Ensure you have **Python 3.8+** installed. It is recommended to use a virtual environment.

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Train the AI Model

**⚠️ Crucial Step:** You must run the training script first. The repository does not include pre-trained models or results to keep the codebase clean.

```bash
python train_model.py
```

**✅ What happens when you run this?**

The script will process the raw data in `data/`, train the Random Forest model, and generate the following files in your root directory:

- `model.pkl`: The saved trained model file
- `predictions.csv`: Model predictions vs. Actual RUL for the test set
- `importance.csv`: Feature importance rankings (which sensors matter most)
- `processed_test_history.csv`: Processed time-series data used for dashboard plotting

### 4. Launch the Dashboard

Once the model is trained, start the visual interface: 

```bash
streamlit run dashboard.py
```

Your browser will automatically open the dashboard at `http://localhost:8501`.

---

## 🖥️ Dashboard Walkthrough

The dashboard is divided into five key sections:

### 1️⃣ Overview & Performance
View high-level model metrics (MAE, RMSE) and dataset details.

### 2️⃣ Data Insights
Explore sensor correlations via heatmaps to understand data redundancy and relationships between sensors.

### 3️⃣ Predictions Analysis
Compare **Predicted RUL vs. Actual RUL** to assess model accuracy across the entire test fleet.

### 4️⃣ Unit Specific Analysis
Deep dive into individual engines. Select a Unit ID to visualize its specific sensor degradation path over time and see how sensor readings evolve as the engine approaches failure.

### 5️⃣ Maintenance Insights (The "Business" Tab)

- **Fleet Status Board:** Instantly see which engines are CRITICAL (<20 cycles), WARNING (<50 cycles), or HEALTHY
- **Business Value:** Estimated cost savings based on preventing failures
- **Feature Importance:** Technical breakdown of which sensors (e.g., Sensor 11 Std Dev) are driving predictions

---

## 🛠️ Technical Details

### Data Preprocessing

**Sensor Selection:** We focus on sensors 2, 3, 4, 7, 11, and 15, which show the strongest correlation with engine degradation in the FD001 dataset. 

**Rolling Windows:** To capture the rate of change, we compute rolling means and standard deviations (window size = 5) for these sensors. This transforms raw sensor readings into features that capture degradation trends.

### Modeling Strategy

**Algorithm:** Random Forest Regressor (Ensemble learning approach)

**RUL Clipping:** The target RUL is clipped at 125 cycles.  This "Piecewise Linear" approach teaches the model that new engines have a constant "healthy" phase before linear degradation begins, significantly improving accuracy. 

**Training Process:**
1. Load and preprocess training data
2. Engineer rolling statistics features
3. Train Random Forest with optimal hyperparameters
4. Validate on test set
5. Generate predictions and feature importance metrics

### Key Metrics

The model is evaluated using:
- **MAE (Mean Absolute Error):** Average prediction error in cycles
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily
- **R² Score:** Proportion of variance explained by the model

---

## 📊 Dataset Information

### NASA C-MAPSS FD001 Dataset

**Source:** [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

**Description:** The dataset contains simulated run-to-failure data for turbofan engines under one operational condition and one fault mode. 

**Data Structure:**
- **Training Set:** 100 engines, complete run-to-failure trajectories
- **Test Set:** 100 engines, truncated at random points before failure
- **Sensors:** 21 sensor channels measuring temperature, pressure, speed, etc. 
- **Operating Conditions:** 3 channels (altitude, throttle, etc.)

**File Format:** Space-delimited text files with columns: 
```
unit_id | time_cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 | ...  | sensor_21
```

---

## 🔍 Understanding the Results

### Feature Importance

The model identifies which sensor readings are most predictive of failure:

- **Sensor 11 (Std Dev):** Often the top predictor, indicating pressure variations
- **Sensor 15 (Mean):** Captures temperature trends
- **Rolling Statistics:** Standard deviations typically outperform raw means, as they capture instability

### Prediction Accuracy

- **Healthy Engines:** Model tends to be highly accurate for engines far from failure
- **Critical Range:** Some uncertainty exists in the final 20-30 cycles
- **Business Impact:** Even with ±10 cycle uncertainty, the system provides actionable early warnings

---

## 💡 Use Cases & Applications

This system can be adapted for: 

✅ **Aviation Maintenance:** Schedule engine overhauls during planned downtime  
✅ **Manufacturing:** Monitor production equipment health  
✅ **Energy Sector:** Predict turbine failures in power plants  
✅ **Automotive Industry:** Fleet management for commercial vehicles  

---

## 🤝 Contributing

This is a course project, but feedback and suggestions are welcome! 

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new visualization'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 License

This project is created for educational purposes as part of CSC4704 coursework.  The NASA C-MAPSS dataset is publicly available for research use.

---

## 👥 Team Members

**Course:** CSC4704 - IoT and Big Data Analytics  
**Institution:** [Your University Name]  
**Semester:** [Semester/Year]

---

## 📚 References

1.  Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set.* NASA Ames Prognostics Data Repository.
2. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.
3. Heimes, F. O.  (2008). *Recurrent neural networks for remaining useful life estimation.* PHM Conference.

---

## 🐛 Troubleshooting

### Common Issues

**Error: `FileNotFoundError: data/train_FD001.txt not found`**
- Ensure the dataset files are placed in the `data/` directory
- Download from NASA repository if missing

**Error: `ModuleNotFoundError: No module named 'streamlit'`**
- Run `pip install -r requirements.txt` again
- Verify your virtual environment is activated

**Dashboard shows no data**
- Run `train_model.py` first to generate required files
- Check that `predictions.csv` and `model.pkl` exist in the root directory

**Poor model performance**
- Verify data integrity in `data/` folder
- Check that all 21 sensor columns are present in input files

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the team through the course portal.

---

**⭐ If you find this project helpful, please star the repository!**