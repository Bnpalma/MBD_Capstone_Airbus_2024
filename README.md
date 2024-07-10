# Fuel Leak Detection in Aircraft

This repository contains the implementation of a project focused on detecting fuel leaks in aircraft using advanced data analytics and machine learning techniques. The goal is to improve the safety, reliability, and efficiency of aircraft operations by providing early detection of fuel leaks.

## Project Overview

Fuel leaks in aircraft can pose significant safety risks and lead to substantial operational disruptions and costs. Traditional detection methods rely on visual inspections and on-board sensors that only detect large leaks. This project leverages machine learning to detect even small leaks early, thereby mitigating risks and reducing costs.

## Repository Contents

- `script_EDA.py`: This script contains the code for the Exploratory Data Analysis (EDA) of the dataset as functions.
- `LSTM Autoencoders.ipynb`: This Jupyter Notebook details the implementation of LSTM autoencoders for anomaly detection in the flight data.
- `EDA and Processing.ipynb`: This notebook includes the comprehensive EDA and data processing steps, including feature engineering and outlier handling.
- `XGBoost.ipynb`: This notebook contains the implementation of the XGBoost model used for supervised learning and leak detection.
- `Leak implanting.ipynb`: This notebook explains the process of synthetic leak data generation and implantation for model training and testing.

## Methodology

1. **Data Preparation and Processing**:
   - Initial data cleaning and preprocessing.
   - Handling missing values and outliers.
   - Feature engineering to create new variables for better prediction.

2. **Exploratory Data Analysis (EDA)**:
   - Detailed analysis of the dataset to uncover underlying patterns.
   - Visualization of key metrics and relationships.

3. **Model Implementation**:
   - **XGBoost**: A supervised learning model was trained to predict the presence of fuel leaks. The model was optimized using cross-validation and hyper-parameter tuning.
   - **LSTM Autoencoders**: Anomaly detection using LSTM autoencoders was employed to identify unusual patterns in the flight data.

4. **Leak Implantation**:
   - Synthetic leaks were implanted in the dataset to simulate real-world scenarios and evaluate model performance.

## Key Results

- The XGBoost model achieved an ROC-AUC of 81.84%, with an accuracy of 77.82%, precision of 81.93%, and recall of 71.64%.
- The LSTM autoencoder effectively detected anomalies in the data, demonstrating the potential for early leak detection.

## Future Improvements

- **Enhanced Computational Power**: Utilize more powerful hardware for model training.
- **Integration of Additional Data Sources**: Incorporate real-time weather, flight paths, and maintenance logs.
- **Real-time Implementation and Testing**: Deploy and test models in real-time operational environments.
- **Reduction of False Positives**: Focus on reducing the rate of false positives to minimize operational disruptions.

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bnpalma/MBD_Capstone_Airbus_2024.git
2. **Explore the notebooks**
