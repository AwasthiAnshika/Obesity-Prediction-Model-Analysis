# Obesity Prediction Model Analysis

A comprehensive analysis of machine learning models to predict **obesity levels** from lifestyle, demographic, and health-related features using a structured ML pipeline. This repository contains Jupyter notebooks for training, evaluating, and comparing multiple classification models.

## Project Overview
Obesity is a critical global health concern. Early and accurate prediction can help with intervention and personalized recommendations. This project builds and compares machine learning models to classify individuals into obesity categories using a publicly available dataset.

## Repository Structure
Obesity-Prediction-Model-Analysis
│
├── DT.ipynb

│   └── Decision Tree model implementation and evaluation

│
├── KNN.ipynb

│   └── K-Nearest Neighbors model with feature scaling

│
├── Naive_Bayes.ipynb

│   └── Naive Bayes classifier implementation

│
├── Random_Forest.ipynb

│   └── Ensemble learning using Random Forest

│
├── README.md

│   └── Project documentation

│
└── LICENSE

    └── License information

## Machine Learning Pipeline
The workflow implemented across notebooks follows a standard supervised learning pipeline:
### 1. **Data Loading**
- Import the raw obesity dataset (CSV) containing features such as age, height, weight, eating habits, physical activity, etc.
- Target labels represent multiple obesity levels.
### 2. **Exploratory Data Analysis (EDA)**
- Analyze feature distributions and data balance.
- Identify missing values and understand relationships among features.
### 3. **Preprocessing**
- **Handling Missing Values** – Impute or drop missing entries as needed.
- **Encoding Categorical Features** – Convert label or category features to numeric (e.g., one-hot encoding or label encoding).
- **Feature Scaling / Normalization** – Standardize or normalize numeric features for models sensitive to scale (e.g., KNN).
- **Train/Test Split** – Split dataset into training and test sets (common split like 80/20).
  
These steps ensure data quality and compatibility with machine learning models. :contentReference[oaicite:0]{index=0}

## Models Implemented
This project includes the following supervised classification algorithms:

| Model | Notebook |
|-------|----------|
| **Decision Tree** | `DT.ipynb` |
| **Random Forest** | `Random_Forest.ipynb` |
| **K-Nearest Neighbors (KNN)** | `KNN.ipynb` |
| **Naive Bayes** | `Naive_Bayes.ipynb` |

Each notebook follows a consistent approach:
- Train model on training data
- Predict on test data
- Record evaluation results
- Visualize performance
  
## Evaluation Metrics
Model performance is evaluated using the following metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall percentage of correctly classified instances. |
| **Precision** | Correct positive predictions / total predicted positives. |
| **Recall (Sensitivity)** | Correct positive predictions / total actual positives. |
| **F1-Score** | Harmonic mean of precision and recall (balances them). |
| **Confusion Matrix** | Breakdown of classification results for each class. |

A dedicated notebook or script (`F1_score.ipynb`) is used to consolidate and visualize all metrics for easy comparison across classifiers.
