# End-to-End Credit Card Fraud Detection
data source : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

This project is an **end-to-end machine learning pipeline** for detecting credit card fraud. It includes **data preprocessing, model training, evaluation, and an interactive Streamlit app** for predictions.

---

## 🚀 Project Overview

Credit card fraud is a highly imbalanced problem. This project tackles it with the following key features:

- **Data Imbalance Handling**: Over-sampling, under-sampling, and custom sampling strategies.
- **Models**: XGBoost & Random Forest for robust fraud prediction.
- **Custom Threshold Selection**: Optimized for F1-score to reduce false negatives.
- **Scalable Architecture**: Modular functions, OOP design, and config-driven workflow.
- **Logging System**: Tracks all training and evaluation steps.
- **Interactive App**: Streamlit app to upload transaction CSVs and get labeled predictions.

---

## 📂 Folder Structure
fraud_detection/
│── config/ # YAML configuration files
│── data/ # Dataset (e.g., creditcard.csv)
│── notebooks/
│ └── oversampling.ipynb
│ └── undersampling.ipynb
│ └── custom_sampling.ipynb
│ └── bestmodel.ipynb
│── EDA.ipynb
│── app.py #streamlit

│── src/
│ ├── train.py # Training pipeline
│ ├── preprocessing.py # Preprocessing & feature engineering
│ ├── utils.py # Utility functions (logging, config, seed)
│ └── model.py # modeling
│ └── features.py
│ └── evaluation.py 

│── logs/ # Log files
│── models/ # Trained models
│── tests/ # Unit tests
   └── preprocessing_test.py 
│── requirements.txt # Python dependencies
