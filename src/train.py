"""
AML Detector — Improved Training
Catches more laundering than the original notebook by:
- Using all 9,873 fraud cases (vs 114) from the full dataset
- Adding behavioral features (sender frequency, time of day, "just below 10k", etc.)
- Using class_weight='balanced' instead of SMOTE (works better with so few real fraud cases)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sns.set_theme(style="darkgrid")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "SAML-D.csv")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
NORMAL_SAMPLE_SIZE = 500_000


def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Full dataset: {len(df):,} rows, {df['Is_laundering'].sum():,} fraud cases")

    # Keep all fraud + a sample of normal transactions
    fraud = df[df["Is_laundering"] == 1]
    normal = df[df["Is_laundering"] == 0].sample(n=NORMAL_SAMPLE_SIZE, random_state=42)
    df = pd.concat([fraud, normal]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Working set: {len(df):,} rows ({df['Is_laundering'].mean():.2%} fraud)")
    return df


def add_features(df):
    print("Adding features...")

    # Amount features
    df["log_amount"] = np.log1p(df["Amount"])
    df["just_below_10k"] = ((df["Amount"] > 9000) & (df["Amount"] < 10000)).astype(int)
    df["is_round_amount"] = (df["Amount"] % 1000 == 0).astype(int)

    # Time features
    df["hour"] = df["Time"].str[:2].astype(int)
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] < 6)).astype(int)

    # Geo / currency
    df["cross_border"] = (df["Sender_bank_location"] != df["Receiver_bank_location"]).astype(int)
    df["currency_mismatch"] = (df["Payment_currency"] != df["Received_currency"]).astype(int)

    # Behavioral / network features
    df["sender_freq"] = df.groupby("Sender_account")["Sender_account"].transform("count")
    df["receiver_freq"] = df.groupby("Receiver_account")["Receiver_account"].transform("count")
    df["sender_avg_amount"] = df.groupby("Sender_account")["Amount"].transform("mean")
    df["amount_vs_sender_avg"] = df["Amount"] / df["sender_avg_amount"].replace(0, 1)

    # Payment type one-hot
    payment_dummies = pd.get_dummies(df["Payment_type"], prefix="pay").astype(int)
    df = pd.concat([df, payment_dummies], axis=1)

    features = [
        "Amount", "log_amount", "just_below_10k", "is_round_amount",
        "hour", "is_night",
        "cross_border", "currency_mismatch",
        "sender_freq", "receiver_freq", "amount_vs_sender_avg",
    ] + list(payment_dummies.columns)

    print(f"  {len(features)} features ready")
    return df, features


def train_and_evaluate(df, features):
    X = df[features]
    y = df["Is_laundering"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train fraud: {y_train.sum()} | Test fraud: {y_test.sum()}")

    print("\nTraining Random Forest (class_weight='balanced')...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("\nResults:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance plot
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.head(15).values, y=importances.head(15).index)
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"\nFeature importance plot saved to {out_path}")

    return model


def main():
    df = load_data()
    df, features = add_features(df)
    train_and_evaluate(df, features)


if __name__ == "__main__":
    main()
