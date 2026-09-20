"""
Step: Explore the GTFS static data we downloaded earlier.

This doesn't do anything fancy yet -- it's just about getting comfortable
reading and summarizing real-world data with pandas, since we'll need the
same skills soon for the actual delay-prediction model.

Run from the project root:
    python scripts/explore_gtfs.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_gtfs"

# GTFS route_type codes -> human-readable names (from the official GTFS spec)
ROUTE_TYPE_NAMES = {
    0: "Tram/Light rail",
    1: "Subway/U-Bahn",
    2: "Rail (S-Bahn/Regional)",
    3: "Bus",
    4: "Ferry",
    100: "Rail (long distance)",
    700: "Bus",
    900: "Tram",
}


def main():
    routes = pd.read_csv(DATA_DIR / "routes.txt")
    stops = pd.read_csv(DATA_DIR / "stops.txt")
    trips = pd.read_csv(DATA_DIR / "trips.txt")

    print("=" * 50)
    print("BASIC COUNTS")
    print("=" * 50)
    print(f"Total routes: {len(routes)}")
    print(f"Total stops:  {len(stops)}")
    print(f"Total trips:  {len(trips)}")

    print("\n" + "=" * 50)
    print("ROUTES BY TYPE (bus, tram, subway, etc.)")
    print("=" * 50)
    routes["type_name"] = routes["route_type"].map(ROUTE_TYPE_NAMES).fillna("Other")
    print(routes["type_name"].value_counts())

    print("\n" + "=" * 50)
    print("HOW MANY TRIPS DOES EACH ROUTE RUN? (top 10 busiest routes)")
    print("=" * 50)
    trips_per_route = trips.groupby("route_id").size().sort_values(ascending=False).head(10)
    # Join with route short names so it's readable, not just IDs
    busiest = trips_per_route.reset_index(name="trip_count").merge(
        routes[["route_id", "route_short_name", "type_name"]], on="route_id"
    )
    print(busiest[["route_short_name", "type_name", "trip_count"]].to_string(index=False))

    print("\n" + "=" * 50)
    print("SAMPLE STOPS")
    print("=" * 50)
    print(stops[["stop_name", "stop_lat", "stop_lon"]].sample(5, random_state=1).to_string(index=False))


if __name__ == "__main__":
    main()