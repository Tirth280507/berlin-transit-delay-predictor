"""
Check whether our collected delay_summary.csv has good enough coverage
to start training a model on, or if we should keep collecting longer.

Run from the project root:
    python scripts/check_coverage.py
"""

import pandas as pd
from pathlib import Path

SUMMARY_FILE = Path(__file__).resolve().parent.parent / "data" / "delay_summary.csv"


def main():
    df = pd.read_csv(SUMMARY_FILE)

    print(f"Total rows: {len(df)}")
    print(f"Unique routes seen: {df['route_id'].nunique()}")

    print("\n--- Days of week covered ---")
    print(df["day_of_week"].value_counts())

    print("\n--- Hours covered (0-23) ---")
    hours_seen = sorted(df["hour"].unique())
    print(f"Hours present: {hours_seen}")
    missing_hours = [h for h in range(24) if h not in hours_seen]
    print(f"Hours MISSING entirely: {missing_hours if missing_hours else 'none'}")

    print("\n--- How trustworthy are the samples? ---")
    print(f"Rows with fewer than 5 samples (shaky data): {(df['sample_count'] < 5).sum()}")
    print(f"Rows with 5-20 samples (okay): {((df['sample_count'] >= 5) & (df['sample_count'] < 20)).sum()}")
    print(f"Rows with 20+ samples (solid): {(df['sample_count'] >= 20).sum()}")


if __name__ == "__main__":
    main()