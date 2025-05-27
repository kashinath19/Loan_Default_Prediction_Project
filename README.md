# 💳 Predicting Loan Defaulters with Machine Learning 🧠
Welcome to the Predicting Loan Defaulters with Machine Learning Models for Credit Card Management project! 🚀 This repository contains a machine learning solution to identify potential loan defaulters using credit card transaction data, aimed at improving credit risk management.

## 📖 Project Overview
This project uses Naive Bayes (NBC) and K-Nearest Neighbors (KNN) classifiers to predict loan defaulters based on transaction data. It includes data preprocessing, model training, and evaluation, with visualizations to understand fraud patterns. 📊


## 🔍 Problems Identified

Fraud Detection Challenges 🕵️: Manual fraud detection in credit card transactions is slow and prone to errors.
Class Imbalance ⚖️: The dataset has a significant imbalance, with very few fraud cases (e.g., 958 fraud vs. 96,296 non-fraud), leading to biased models.
Inefficient Risk Management 📉: Traditional methods struggle to predict loan defaults accurately, impacting financial decision-making.
Lack of Automation 🤖: Manual processes hinder real-time fraud detection and credit risk assessment.

## 🛠️ Problems Solved

Automated Fraud Detection 🔍: Implemented NBC and KNN classifiers to automate the identification of fraudulent transactions.
Handled Class Imbalance ⚖️: Visualized class distribution with a count plot, highlighting the imbalance for better model tuning.
Improved Prediction Accuracy 🎯: Achieved high accuracy with KNN (99.16%) and NBC (98.93%) through effective preprocessing and model training.
Accessible Insights 📈: Used visualizations like count plots and confusion matrices to provide clear insights into fraud patterns.

## 🏆 Achievements

High Model Performance 🌟: KNN achieved 99.16% accuracy, 90.02% precision, and 70.71% recall, while NBC scored 98.93% accuracy.
Effective Preprocessing 🧹: Encoded categorical features (e.g., merchant, category, gender) using LabelEncoder and handled large datasets (97,254 entries).
Robust Evaluation 📊: Calculated precision, recall, F1-score, and confusion matrices to thoroughly assess model performance.
Practical Application 💼: Built a tool that financial institutions can use to predict loan defaulters, enhancing credit risk management.

## 🚀 Features

📂 Dataset Loading: Load credit card transaction data from CSV (fraudTrain.csv).
🧹 Preprocessing: Encode categorical variables and prepare data for model training.
📈 Model Training: Train NBC and KNN classifiers to predict fraud.
📊 Evaluation: Compute accuracy, precision, recall, F1-score, and visualize results with confusion matrices.
💾 Model Persistence: Save trained models using joblib for future use.

## 🛠️ Technologies Used

Python 🐍: Core programming language.
Pandas/NumPy 📚: For data manipulation.
Scikit-learn 📊: For machine learning models (NBC, KNN) and metrics.
Matplotlib/Seaborn 📉: For data visualization.
Joblib 💾: For model saving and loading.

## 📋 Requirements

Software: Python 3.7+, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib.
Hardware: Minimum 4GB RAM, 250GB storage.

## 🖥️ How to Run

Clone the Repository:
git clone https://github.com/yourusername/loan-defaulter-prediction.git
cd loan-defaulter-prediction


Install Dependencies:
pip install -r requirements.txt


Run the Notebook:

Open Main.ipynb in Jupyter Notebook or Google Colab.
Execute the cells to preprocess data, train models, and evaluate results.



## 📊 Results

KNN Classifier: Accuracy: 99.16%, Precision: 90.02%, Recall: 70.71%.
Naive Bayes Classifier: Accuracy: 98.93%, Precision: 49.46%, Recall: 49.73%.
Visualizations include count plots for class distribution and confusion matrices for model performance.

## 🌟 Future Enhancements

Address class imbalance using techniques like SMOTE. ⚖️
Explore ensemble methods (e.g., Random Forest) for better performance. 🌲
Add real-time transaction monitoring with API integration. 📡

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

Happy credit risk management with AI! 💸✨
