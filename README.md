# Loan Default Prediction System Using Machine Learning

## Overview

This project is an end-to-end Machine Learning system designed to predict whether a borrower is likely to default on a loan using real-world financial data from the LendingClub dataset. The system applies advanced preprocessing, feature engineering, imbalance handling, model optimization, and deployment techniques to build a realistic credit-risk prediction pipeline.

The primary goal of this project is to assist financial institutions in identifying high-risk borrowers before loan approval, reducing financial losses while improving lending decisions.

The final application is deployed using Streamlit on Hugging Face Spaces with an interactive user interface that allows users to input borrower details and receive real-time loan risk predictions.

---

# Live Demo

🚀 **Deployed Application:**
[Loan Default Prediction App](https://huggingface.co/spaces/hacker108/loan-default-prediction?utm_source=chatgpt.com)

---

# Problem Statement

Loan default prediction is one of the most important applications of Machine Learning in the banking and financial sector. Financial institutions must evaluate the risk associated with approving loans to borrowers. Incorrect approvals can lead to significant financial losses, while overly restrictive lending policies can reduce business opportunities.

This project aims to solve this classification problem by building a predictive model capable of estimating the probability of borrower default using historical financial and credit-related information.

---

# Dataset Information

Dataset Used:

* LendingClub Loan Dataset

The dataset contains real-world borrower and loan information including:

* Loan Amount
* Interest Rate
* Installment Amount
* Employment Length
* Annual Income
* Home Ownership
* Verification Status
* Loan Purpose
* Debt-to-Income Ratio (DTI)
* Open Credit Accounts
* Revolving Balance
* Revolving Utilization
* FICO Score
* Total Accounts
* Inquiry History
* Loan Status (Target Variable)

---

# Machine Learning Workflow

## 1. Data Cleaning & Preprocessing

The dataset was cleaned and prepared before training the models.

### Tasks Performed

* Removed unnecessary columns
* Handled missing values
* Cleaned employment length feature
* Converted categorical features into numerical format
* Created a binary target variable for loan default prediction
* Generated a combined FICO score

---

# Exploratory Data Analysis (EDA)

Extensive EDA was performed to understand:

* Class imbalance
* Loan distributions
* Financial risk patterns
* Correlation between features
* Outlier behavior
* Borrower characteristics

### Visualizations Included

* Loan Status Distribution
* Interest Rate Distribution
* Annual Income Distribution
* Correlation Heatmap
* Loan Amount Outlier Detection
* Default Pattern Analysis

Libraries Used:

* Matplotlib
* Seaborn

---

# Feature Engineering

Feature engineering was one of the most important parts of this project. Several financial risk-related features were created to improve model learning capability.

## Engineered Features

### Income-to-Loan Ratio

income_to_loan = \frac{annual_inc}{loan_amnt}

Measures repayment capability relative to loan amount.

---

### Installment Burden Ratio

installment_ratio = \frac{installment}{annual_inc}

Represents financial burden caused by monthly installment.

---

### Credit Load

credit_load = \frac{revol_bal}{annual_inc}

Shows revolving debt pressure relative to income.

---

### Risk Interaction Score

risk_interaction = dti \times int_rate

Captures combined financial stress.

---

# Handling Imbalanced Dataset

The dataset was highly imbalanced:

* Majority borrowers repaid loans
* Minority borrowers defaulted

Using accuracy alone would produce misleading results.

## Techniques Used

* SMOTE (Synthetic Minority Oversampling Technique)
* Threshold Optimization
* Stratified K-Fold Cross Validation
* Class Weighting

These techniques improved the model’s ability to detect minority default cases effectively.

---

# Models Used

## 1. Logistic Regression

Used as the baseline linear classification model.

### Advantages

* Fast
* Interpretable
* Good baseline model

---

## 2. Random Forest

Used for ensemble-based nonlinear learning.

### Advantages

* Handles nonlinear patterns
* Robust against overfitting
* Good feature importance analysis

---

## 3. XGBoost

Used as the final optimized model.

### Advantages

* High predictive performance
* Excellent handling of tabular data
* Strong performance on imbalanced datasets
* GPU acceleration support

---

# Model Evaluation Metrics

Since the dataset was imbalanced, multiple evaluation metrics were used instead of relying only on accuracy.

## Metrics Used

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* Precision-Recall Curve

---

# Final Model Performance

| Model               | F1 Score |
| ------------------- | -------- |
| Logistic Regression | 0.445    |
| Random Forest       | 0.448    |
| XGBoost             | 0.456    |

### Best Model

✅ XGBoost achieved the best overall performance and provided the best balance between recall and precision.

---

# Why F1 Score Was Preferred

The dataset was imbalanced, meaning non-defaulters significantly outnumbered defaulters.

In such cases:

* Accuracy can be misleading
* Precision and Recall become more important

The F1 Score provides a balanced evaluation of:

* False Positives
* False Negatives

making it more suitable for financial risk prediction problems.

---

# Deployment

The final model was deployed using:

* Streamlit
* Hugging Face Spaces

## Application Features

* Interactive UI
* Real-time loan prediction
* Default probability estimation
* Risk level analysis
* Financial ratio analysis

---

# User Interface Features

The deployed application includes:

* Loan risk prediction
* Probability visualization
* Financial analysis metrics
* Risk categorization
* User-friendly fintech-style interface

---

# Technologies Used

## Programming Language

* Python

## Data Analysis

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)

## Deployment

* Streamlit
* Hugging Face Spaces

## Model Saving

* Joblib

---

# Key Learnings

Through this project, I gained hands-on experience in:

* End-to-end Machine Learning workflow
* Financial risk prediction systems
* Data preprocessing
* Feature engineering
* Imbalanced classification handling
* Cross-validation techniques
* Threshold optimization
* Ensemble learning
* Model deployment
* Building interactive ML applications

One of the most important learnings from this project was understanding that improving data quality and feature engineering often has a greater impact than simply changing algorithms.

---

# Future Improvements

Possible future enhancements include:

* SHAP Explainability
* Advanced Hyperparameter Tuning
* Optuna Integration
* Real-time API Deployment
* Improved UI/UX Design
* PDF Report Generation
* User Authentication
* Cloud Database Integration
* Deep Learning Models
* Real-time Banking Integration

---

# Project Structure

```text
loan-default-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── loan_default_xgboost.pkl
├── feature_columns.pkl
└── loan_default_prediction.ipynb
```

---

# Conclusion

This project demonstrates a complete real-world Machine Learning pipeline for solving a financial risk prediction problem. By combining feature engineering, imbalance handling, threshold optimization, ensemble learning, and deployment techniques, the final system provides a realistic and practical loan default prediction solution.

The project highlights the importance of data understanding, preprocessing, and business-oriented thinking in building effective Machine Learning systems for real-world applications.

---

# Author

**Arpit Yadav**
B.Tech Electrical Engineering Student
Machine Learning & AI Enthusiast

GitHub Repository:
[GitHub Repository](https://github.com?utm_source=chatgpt.com)
