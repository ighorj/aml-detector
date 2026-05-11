# AML Detector 🔍

A machine learning pipeline for detecting money laundering patterns in financial transactions.

Built as a beginner ML project, trained on the SAML-D dataset (28 real-world AML typologies) 
using Random Forest with SMOTE to handle class imbalance.

---

## 🔎 What it does

- Explores 100k financial transactions from the SAML-D dataset
- Engineers features to detect smurfing and layering patterns
- Handles severe class imbalance (0.1% fraud rate) with SMOTE
- Trains a Random Forest classifier to flag suspicious transactions
- Visualizes feature importance and laundering type distribution

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 98% |
| Recall (laundering) | 33% |
| Fraud cases in dataset | 114 out of 100k transactions |

> Accuracy is high due to class imbalance — 99.9% of transactions are normal.
> Recall of 33% means the model catches 1 in 3 real laundering cases.
> This is a known challenge in AML detection and a next step for improvement.

---

## 📈 Key Findings

- `receiver_frequency` and `Amount` are the strongest predictors of laundering
- `amount_vs_sender_avg` captures unusual behavior relative to a sender's own history
- Structuring and Cash Withdrawal are the most common laundering typologies in the dataset
- Cross-border transactions show slight correlation with suspicious activity

---

## 🧰 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, Imbalanced-learn
- Matplotlib, Seaborn
- Jupyter Notebooks

---

## 📦 Data

Download the SAML-D dataset from Kaggle and place it in the `data/` folder:  
🔗 https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml

---

## 🗂️ Project Structure