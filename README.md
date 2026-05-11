# AML Detector 🔍

A machine learning pipeline for detecting money laundering patterns in financial transactions.

Trained on the SAML-D dataset (28 real-world AML typologies) using a Random Forest classifier with balanced class weights to handle severe class imbalance.

---

## 📌 Overview

Money laundering is the process of making illegally obtained money appear legitimate. It typically happens in three stages:

- **Placement** — introducing dirty money into the financial system (e.g. cash deposits)
- **Layering** — disguising the trail through complex transactions (e.g. cross-border transfers, smurfing)
- **Integration** — reintroducing the money as legitimate funds

This project builds a machine learning classifier that flags suspicious financial transactions based on behavioral patterns — mimicking what real AML compliance teams do at banks and fintechs.

---

## 🔎 What it does

- Loads the full SAML-D dataset (9.5M transactions, 9,873 fraud cases)
- Samples 500k normal transactions + all fraud cases for training
- Engineers 18 behavioral features (time, amount, currency, sender/receiver patterns)
- Trains a Random Forest with `class_weight='balanced'` to handle the imbalance
- Evaluates on a stratified holdout set using Precision, Recall and F1-score
- Visualizes feature importance

---

## 📊 Results

| Metric | Score |
|---|---|
| Recall (laundering) | **78%** |
| Precision (laundering) | **37%** |
| F1-score (laundering) | **0.50** |
| Fraud cases detected | **1,548 out of 1,975** in test set |
| Training samples | 407,898 (7,898 fraud) |
| Test samples | 101,975 (1,975 fraud) |

> **Why not accuracy?** Accuracy is misleading on imbalanced datasets. A model predicting "Normal" for every transaction would get 98%+ accuracy while catching zero fraud. **Recall** (catch rate) and **F1** (balance between catching fraud and minimizing false alarms) are what matter in AML.

> **Progress vs previous version:** This iteration uses **all 9,873 fraud cases** (vs 114 before) and added behavioral features like time-of-day, log amount, and "just below R$10k" flags. Recall jumped from 33% → 78%, F1 from 0.05 → 0.50.

---

## 📈 Key Findings

- `receiver_freq` — accounts that receive from many different senders are a classic *laranja* (money mule) signal.
- `sender_freq` — high-velocity senders are disproportionately represented in laundering cases.
- `amount_vs_sender_avg` captures behavior that deviates from the sender's own history — a stronger signal than raw amount.
- `log_amount` outperforms raw `Amount` because laundering amounts span several orders of magnitude.
- `just_below_10k` flag catches **structuring** — deliberately splitting transactions just below reporting thresholds.
- Cross-border transactions and currency mismatches show weak but present correlation.

---

## 🧠 Feature Engineering

| Feature | Description | AML Signal |
|---|---|---|
| `Amount`, `log_amount` | Raw and log-scaled transaction amount | Captures both large transfers and structured small ones |
| `is_round_amount` | Amount divisible by 1000 | Round amounts are a structuring signal |
| `just_below_10k` | Amount between $9,000–$10,000 | Classic threshold-avoidance pattern |
| `hour`, `is_night` | Transaction hour and night flag (00:00–06:00) | Off-hours activity is a risk indicator |
| `cross_border` | Sender and receiver in different countries | Cross-border layering |
| `currency_mismatch` | Payment and received currency differ | Currency conversion used to obscure trails |
| `sender_freq`, `receiver_freq` | Activity counts per account | Mules and high-velocity accounts |
| `sender_avg_amount` | Sender's historical average | Baseline for anomaly detection |
| `amount_vs_sender_avg` | Current amount ÷ sender's average | Detects unusual spikes |
| `pay_*` | One-hot encoded payment type | Captures payment-method patterns |

---

## 🗂️ Project Structure

```
aml-detector/
├── data/
│   ├── SAML-D.csv              # Full dataset (not tracked by git, ~1GB)
│   └── SAML-sample.csv         # 100k row sample (not tracked by git)
├── notebooks/
│   └── 01_exploration.ipynb    # EDA and original beginner pipeline
├── src/
│   └── train.py                # Improved training script
├── outputs/
│   ├── class_distribution.png
│   ├── laundering_types.png
│   ├── amount_distribution.png
│   ├── amount_vs_laundering.png
│   └── feature_importance.png
├── .gitignore
└── README.md
```

---

## 📦 Dataset

**SAML-D — Synthetic Anti-Money Laundering Dataset**
- 28 real-world AML typologies (Smurfing, Structuring, Layered Fan-Out, Scatter-Gather, etc.)
- 9.5M transactions, 9,873 labeled as laundering (0.1% prevalence)
- Labels: `Is_laundering` (0 = normal, 1 = suspicious), `Laundering_type`

🔗 https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml

> The dataset is not tracked by git due to its size. Download it from Kaggle and place it in the `data/` folder.

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data loading, manipulation, feature engineering |
| NumPy | Numerical operations |
| Scikit-learn | Model training and evaluation |
| Matplotlib + Seaborn | Plotting |
| Jupyter Lab | Interactive notebook environment |

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/ighorj/aml-detector.git
cd aml-detector

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Download the dataset from Kaggle and place it in data/SAML-D.csv
# https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml

# 4. Run the improved training script
python src/train.py

# Or explore the original notebook
jupyter lab notebooks/01_exploration.ipynb
```

---

## 🚧 Known Limitations

- No time-window features (e.g. number of transfers per sender in 10-minute windows)
- No network/graph analysis (detecting laranja chains across accounts)
- Default 0.5 classification threshold — tuning could trade recall for precision
- Tree-based gradient boosting (XGBoost/LightGBM) not yet benchmarked

---

## 🔜 Next Steps

- [ ] Add time-window features (transfer velocity, burst detection)
- [ ] Build account-level network graph with NetworkX
- [ ] Compare against XGBoost / LightGBM
- [ ] Tune classification threshold via precision-recall curve
- [ ] Output a per-account risk report with score and reasons

---

## 👤 Author

Built by [@ighorj](https://github.com/ighorj) as a first ML project.
Open to feedback, suggestions and collaboration.
