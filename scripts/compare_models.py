"""
Compare a few different model types on the same data, so we pick the best
one based on real results -- not just a guess.

Models compared:
- Logistic Regression: the simplest baseline, fast, easy to explain
- Random Forest: handles messy/non-linear patterns well
- Gradient Boosting: often the strongest performer on tabular data like this

Run from the project root:
    python scripts/compare_models.py
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "delay_summary.csv"
DELAY_THRESHOLD_SECONDS = 60
MIN_SAMPLE_COUNT = 5


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

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    print(f"{'Model':<22} {'Accuracy':<12} {'F1 Score':<10}")
    print("-" * 44)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print(f"{name:<22} {acc:<12.2%} {f1:<10.3f}")


if __name__ == "__main__":
    main()