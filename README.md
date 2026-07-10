# 🌍 AirPulse AI – Hybrid AQI Prediction System

AirPulse AI is an advanced Air Quality Index (AQI) prediction system that combines **LSTM (Long Short-Term Memory)** and **XGBoost** models to accurately forecast air quality levels. The system also integrates real-time weather information using the **Open-Meteo API** to improve prediction performance.

## 🚀 Overview

Air pollution has become a major environmental and public health concern. Accurate AQI forecasting helps governments, organizations, and individuals take preventive measures to reduce exposure to harmful pollutants.

This project utilizes a hybrid machine learning and deep learning architecture:

- **LSTM** captures temporal patterns in historical AQI data.
- **XGBoost** learns complex feature relationships for precise AQI prediction.
- **Open-Meteo API** provides real-time weather information to enhance forecasting accuracy.

---

## ✨ Features

- Hybrid LSTM + XGBoost Architecture
- Real-Time AQI Prediction
- Weather Data Integration using Open-Meteo API
- Time-Series Forecasting
- Data Preprocessing and Feature Engineering
- Interactive Web Interface
- High Prediction Accuracy (R² Score: 97.42%)

---

## 🛠️ Tech Stack

### Languages
- Python

### Machine Learning & Deep Learning
- TensorFlow
- Keras
- XGBoost
- Scikit-Learn

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib

### Web Development
- Flask
- HTML
- CSS
- JavaScript

### API
- Open-Meteo API

---

## 📊 Dataset

Dataset used:

**Air Quality Data in India**

Source:
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

### Features Used

- PM2.5
- PM10
- NO₂
- SO₂
- CO
- O₃
- AQI
- Date & Time Information
- City Information

---

## 🧠 Model Architecture

```text
AQI Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Scaling
      │
      ▼
Time-Series Sequence Generation
      │
      ▼
LSTM Network
(Deep Feature Extraction)
      │
      ▼
Extracted Features
      │
      ▼
XGBoost Model
      │
      ▼
Final AQI Prediction
```

---

## 🔄 Workflow

### 1. Data Collection
- Historical AQI data
- Pollutant concentration data
- Weather information from Open-Meteo API

### 2. Data Preprocessing
- Missing Value Handling
- Data Cleaning
- Normalization
- Feature Scaling

### 3. Sequence Generation
- Creates sliding windows of previous 24-hour AQI records

### 4. LSTM Training
- Learns temporal pollution patterns
- Extracts deep features from sequential data

### 5. XGBoost Training
- Uses extracted LSTM features
- Generates final AQI predictions

### 6. Real-Time Prediction
- Fetches weather data
- Processes latest AQI information
- Predicts AQI in real time

---

## 📈 Model Performance

| Metric | Value |
|----------|----------|
| MSE | 542.5099 |
| MAE | 11.0402 |
| R² Score | 0.9742 |

### Performance Analysis

- R² Score of **97.42%** indicates excellent predictive capability.
- Low MAE demonstrates strong forecasting accuracy.
- Hybrid architecture outperforms standalone machine learning models.

---

## 📂 Project Structure

```text
AirPulse-AI/
│
├── data/
│   ├── city_day.csv
│   └── city_hour.csv
│
├── models/
│   ├── lstm_model.keras
│   ├── xgboost_model.json
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AirPulse-AI.git

cd AirPulse-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000
```

---

## 📷 Screenshots

### Dashboard
Add dashboard screenshot here.

### AQI Prediction Output
Add prediction screenshot here.

### Real-Time Weather Integration
Add weather dashboard screenshot here.

---

## 🌱 Sustainable Development Goals (SDGs)

This project supports:

- SDG 3 – Good Health and Well-Being
- SDG 11 – Sustainable Cities and Communities
- SDG 13 – Climate Action

---

## 🔮 Future Enhancements

- IoT Sensor Integration
- Multi-Day AQI Forecasting
- Transformer-Based Models
- Mobile Application Development
- Smart City Deployment
- Real-Time Pollution Alerts

---

