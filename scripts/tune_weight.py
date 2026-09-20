"""
Try a few different weight values for the "Delayed" class and compare
precision/recall/F1 side by side, so we pick the best one based on real
numbers instead of guessing.

Run from the project root:
    python scripts/tune_weight.py
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "delay_summary.csv"
DELAY_THRESHOLD_SECONDS = 60
MIN_SAMPLE_COUNT = 5

WEIGHTS_TO_TRY = [1, 2, 3, 4, 5]


def load_and_prepare():
    df = pd.read_csv(DATA_FILE)
    df = df[df["sample_count"] >= MIN_SAMPLE_COUNT].copy()
    df["is_delayed"] = (df["avg_delay_seconds"] > DELAY_THRESHOLD_SECONDS).astype(int)

    route_encoder = LabelEncoder()
    day_encoder = LabelEncoder()
    df["route_encoded"] = route_encoder.fit_transform(df["route_id"])
    df["day_encoded"] = day_encoder.fit_transform(df["day_of_week"])

    features = df[["route_encoded", "day_encoded", "hour"]]
    label = df["is_delayed"]
    return features, label


def main():
    features, label = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        features, label, test_size=0.2, random_state=42, stratify=label
    )

    print(f"{'Weight':<8} {'Precision':<12} {'Recall':<10} {'F1':<8} {'Accuracy':<10}")
    print("-" * 50)

    for weight in WEIGHTS_TO_TRY:
        sample_weights = y_train.map({0: 1, 1: weight})
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        preds = model.predict(X_test)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        accuracy = accuracy_score(y_test, preds)

        print(f"{weight:<8} {precision:<12.2f} {recall:<10.2f} {f1:<8.2f} {accuracy:<10.2%}")


if __name__ == "__main__":
    main()