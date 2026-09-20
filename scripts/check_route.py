"""
Check whether a specific route (e.g. "U7") exists in our route lookup,
and whether it was ever actually captured in the live delay data
(and therefore known to the trained model).

Run from the project root:
    python scripts/check_route.py U7
"""

import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOOKUP_FILE = BASE_DIR / "data" / "routes_lookup.csv"
SUMMARY_FILE = BASE_DIR / "data" / "delay_summary.csv"


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_route.py <search term, e.g. U7>")
        return

    search = sys.argv[1]

    lookup = pd.read_csv(LOOKUP_FILE)
    matches = lookup[lookup["route_short_name"].astype(str).str.fullmatch(search, case=False)]

    if matches.empty:
        print(f"'{search}' not found in the GTFS schedule data at all -- check spelling.")
        return

    print(f"Found in schedule data:\n{matches}\n")

    summary = pd.read_csv(SUMMARY_FILE)
    route_ids = matches["route_id"].tolist()
    captured = summary[summary["route_id"].isin(route_ids)]

    if captured.empty:
        print(f"'{search}' was NEVER captured in the live delay feed -- "
              f"this is a real data coverage gap (VBB's feed likely never reported "
              f"live delays for this route during our collection window). "
              f"That's why it's missing from the app's dropdown.")
    else:
        print(f"'{search}' WAS captured {len(captured)} times. It should appear in the app. "
              f"Sample data:\n{captured.head()}")


if __name__ == "__main__":
    main()