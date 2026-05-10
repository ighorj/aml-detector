# AML Detector 🔍

A machine learning pipeline for detecting money laundering patterns in financial transactions.

Built as a beginner ML project, trained on the SAML-D dataset (28 real-world AML typologies) 
using XGBoost and Random Forest with SMOTE to handle class imbalance.

---

## 🔎 What it does

- Explores and cleans real-world AML transaction data
- Engineers features to detect smurfing and layering patterns
- Handles severe class imbalance with SMOTE
- Trains and compares Random Forest vs XGBoost classifiers
- Outputs a structured risk report flagging suspicious accounts

---

## 🧰 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost, Imbalanced-learn
- Matplotlib, Seaborn
- Jupyter Notebooks

---

## 📦 Data

Download the SAML-D dataset from Kaggle and place it in the `data/` folder:  
🔗 https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml

---

## 🗂️ Project Structure
