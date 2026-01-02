# AIoT Predictive Maintenance - NASA Turbofan Engine Dataset

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=flat)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Viz-3776AB?style=flat)](https://seaborn.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 📋 Project Overview

This project implements an **AI-powered Internet of Things (AIoT)** solution for predictive maintenance of turbofan engines using the NASA C-MAPSS dataset. The system combines deep learning models with IoT sensor data to predict Remaining Useful Life (RUL) of aircraft engines, enabling proactive maintenance and reducing operational costs.

### Key Features

- 🔮 **Predictive Analytics**: LSTM-based deep learning model for RUL prediction
- 📊 **Real-time Monitoring**: IoT sensor data integration and analysis
- 🎯 **High Accuracy**: Achieves strong performance metrics on NASA dataset
- 📈 **Visualization**: Comprehensive data analysis and result visualization
- 🔄 **Scalable Architecture**: Modular design for easy deployment and scaling

## 📁 Project Structure

```
AIoT-Predictive-Maintenance-NASA-Turbofan/
├── data/                      # Dataset files
├── models/                    # Saved models
├── notebooks/                 # Jupyter notebooks
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   └── Model_Development.ipynb
├── src/                       # Source code
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── visualize_results.py
├── results/                   # Output plots and metrics
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # MIT License
```

## 🗂️ Dataset

The project uses the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset:

- **Source**: NASA Prognostics Data Repository
- **Description**: Run-to-failure simulation data from turbofan engines
- **Sensors**: 21 sensor measurements per time cycle
- **Operating Conditions**: 3 operational settings
- **Scenarios**: Multiple fault modes and conditions

### Dataset Structure

```
data/
├── train_FD001.txt    # Training data
├── test_FD001.txt     # Test data
└── RUL_FD001.txt      # Ground truth RUL values
```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AIoT System Architecture               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  IoT Layer          Processing Layer      AI Layer      │
│  ┌──────────┐      ┌──────────────┐    ┌──────────┐   │
│  │ Sensors  │ ───> │ Data Pipeline│───>│ LSTM     │   │
│  │ - Temp   │      │ - Cleaning   │    │ Model    │   │
│  │ - Press  │      │ - Features   │    │          │   │
│  │ - Speed  │      │ - Normalize  │    └──────────┘   │
│  └──────────┘      └──────────────┘          │         │
│                                               v         │
│                                      ┌──────────────┐   │
│                                      │ Predictions  │   │
│                                      │ (RUL Output) │   │
│                                      └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
```

### How to setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nitezio/AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1.git
   cd AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Download NASA C-MAPSS dataset from [NASA Prognostics Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
   - Place files in `data/` directory

## 💻 Usage

### 1. Data Preprocessing

```python
python src/data_preprocessing.py
```

This script:
- Loads raw sensor data
- Performs data cleaning and normalization
- Engineers relevant features
- Splits data into train/validation/test sets

### 2. Model Training

```python
python src/train_model.py
```

Features:
- LSTM-based architecture
- Early stopping and model checkpointing
- Training history visualization
- Model performance evaluation

### 3. Making Predictions

```python
python src/predict.py --model models/best_model.h5 --data data/test_FD001.txt
```

### 4. Visualization

```python
python src/visualize_results.py
```

Generates:
- Training/validation loss curves
- Prediction vs actual RUL plots
- Sensor data correlation heatmaps
- Error distribution analysis

## 📊 Model Architecture

```python
Model: LSTM-based Sequence Predictor
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_1 (LSTM)               (None, 50, 100)           44400     
dropout_1 (Dropout)         (None, 50, 100)           0         
lstm_2 (LSTM)               (None, 50)                30200     
dropout_2 (Dropout)         (None, 50)                0         
dense_1 (Dense)             (None, 50)                2550      
dense_2 (Dense)             (None, 1)                 51        
=================================================================
Total params: 77,201
Trainable params: 77,201
Non-trainable params: 0
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| RMSE | ~18-22 cycles |
| MAE | ~15-18 cycles |
| R² Score | ~0.75-0.85 |
| Training Time | ~15-20 min (GPU) |

## 🔍 Key Components

### Data Preprocessing
- **Normalization**: Min-Max scaling for sensor readings
- **Feature Engineering**: Rolling statistics, degradation indicators
- **Sequence Generation**: Time-window based sequences for LSTM

### Model Features
- **Architecture**: Stacked LSTM layers with dropout
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Mean Squared Error (MSE)
- **Regularization**: Dropout and early stopping

### IoT Integration
- Real-time sensor data ingestion
- Stream processing capabilities
- Alert system for critical RUL thresholds

## 🎯 Future Enhancements

- [ ] Multi-engine parallel prediction
- [ ] Real-time dashboard with Flask/Streamlit
- [ ] Transfer learning for different engine types
- [ ] Integration with cloud IoT platforms (AWS IoT, Azure IoT)
- [ ] Ensemble methods (LSTM + CNN)
- [ ] Uncertainty quantification
- [ ] Mobile application for maintenance alerts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Nitezio** - *Initial work* - [GitHub Profile](https://github.com/Nitezio)

## 🙏 Acknowledgments

- NASA Prognostics Center of Excellence for providing the C-MAPSS dataset
- CSC4704-1 Course Staff and Faculty
- TensorFlow and Keras teams for excellent deep learning frameworks
- Open-source community for various tools and libraries

## 📧 Contact

Project Link: [https://github.com/Nitezio/AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1](https://github.com/Nitezio/AIoT-Predictive-Maintenance-NASA-Turbofan_for_CSC4704-1)

---

**Note**: This project is developed for educational purposes as part of the CSC4704-1 course curriculum.

## 📚 References

1. Saxena, A., & Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository
2. Zheng, S., et al. (2017). "Long Short-Term Memory Network for Remaining Useful Life estimation"
3. Babu, G. S., et al. (2016). "Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life"

---

⭐ **Star this repository** if you find it helpful!
