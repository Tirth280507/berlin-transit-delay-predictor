"""
Export a small routes_lookup.csv (route_id -> friendly name) so the
Streamlit app can show real route names/numbers instead of cryptic IDs
like "17325_700". This file is tiny and safe to commit, unlike the full
raw_gtfs folder (which is gitignored for being too large).

Run from the project root:
    python scripts/export_route_lookup.py
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_gtfs"
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "routes_lookup.csv"

ROUTE_TYPE_NAMES = {
    0: "Tram", 1: "U-Bahn", 2: "S-Bahn/Regional", 3: "Bus",
    4: "Ferry", 100: "Regional Rail", 400: "U-Bahn", 700: "Bus", 900: "Tram",
}


def main():
    routes = pd.read_csv(RAW_DIR / "routes.txt")
    routes["type_name"] = routes["route_type"].map(ROUTE_TYPE_NAMES).fillna("Other")

    lookup = routes[["route_id", "route_short_name", "type_name"]].copy()
    lookup["display_name"] = lookup["type_name"] + " " + lookup["route_short_name"].astype(str)

    lookup.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(lookup)} routes to {OUT_FILE}")


if __name__ == "__main__":
    main()