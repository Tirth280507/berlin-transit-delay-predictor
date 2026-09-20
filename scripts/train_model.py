"""
Train a model to predict delay RISK (likely delayed vs. likely on time)
for a given route, day of week, and hour.

Steps:
1. Load our collected data/delay_summary.csv
2. Drop rows with too few samples (unreliable/noisy)
3. Create the label: is_delayed = 1 if avg_delay_seconds > DELAY_THRESHOLD_SECONDS
4. Encode route_id and day_of_week as numbers (models need numbers, not text)
5. Train a Gradient Boosting classifier (won a comparison against Logistic
   Regression and Random Forest -- see compare_models.py)
6. Print how well it performed
7. Save the trained model + encoders so the Streamlit app can use them later

Run from the project root:
    python scripts/train_model.py
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "delay_summary.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "model"

# A route/time slot counts as "high risk" if its average delay is over this many seconds.
# 60 seconds = 1 minute. Change this number to make the label stricter or looser.
DELAY_THRESHOLD_SECONDS = 60

# Drop rows we don't trust yet (too few real observations).
MIN_SAMPLE_COUNT = 5


def load_and_prepare():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows")

    df = df[df["sample_count"] >= MIN_SAMPLE_COUNT].copy()
    print(f"Kept {len(df)} rows after dropping low-sample (unreliable) rows")

    df["is_delayed"] = (df["avg_delay_seconds"] > DELAY_THRESHOLD_SECONDS).astype(int)
    print(f"\nLabel balance:\n{df['is_delayed'].value_counts()}")

    return df


def encode_features(df):
    route_encoder = LabelEncoder()
    day_encoder = LabelEncoder()

    df["route_encoded"] = route_encoder.fit_transform(df["route_id"])
    df["day_encoded"] = day_encoder.fit_transform(df["day_of_week"])

    features = df[["route_encoded", "day_encoded", "hour"]]
    label = df["is_delayed"]

    return features, label, route_encoder, day_encoder


def train_and_evaluate(features, label):
    X_train, X_test, y_train, y_test = train_test_split(
        features, label, test_size=0.2, random_state=42, stratify=label
    )

    # Gradient Boosting doesn't have a class_weight option like Random Forest does,
    # so we manually give more importance to "Delayed" examples during training.
    # This trades some precision for better recall -- i.e. it will flag more
    # possible delays, at the cost of occasionally being wrong. For a commuter
    # app, missing a real delay is worse than an unnecessary warning.
    sample_weights = y_train.map({0: 1, 1: 2})

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy on unseen test data: {accuracy:.2%}")
    print("\nDetailed report:")
    print(classification_report(y_test, predictions, target_names=["On time", "Delayed"]))

    return model


def save_everything(model, route_encoder, day_encoder):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "delay_model.joblib")
    joblib.dump(route_encoder, MODEL_DIR / "route_encoder.joblib")
    joblib.dump(day_encoder, MODEL_DIR / "day_encoder.joblib")
    print(f"\nSaved model + encoders to {MODEL_DIR}")


if __name__ == "__main__":
    df = load_and_prepare()
    features, label, route_encoder, day_encoder = encode_features(df)
    model = train_and_evaluate(features, label)
    save_everything(model, route_encoder, day_encoder)