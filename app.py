"""
Berlin Transit Delay Predictor -- Streamlit app.

Loads the trained model + encoders and lets a user pick a real route,
day of week, and hour, then shows the predicted delay risk.

Run from the project root:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "data" / "model"
ROUTES_FILE = BASE_DIR / "data" / "routes_lookup.csv"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_resource
def load_model_and_encoders():
    model = joblib.load(MODEL_DIR / "delay_model.joblib")
    route_encoder = joblib.load(MODEL_DIR / "route_encoder.joblib")
    day_encoder = joblib.load(MODEL_DIR / "day_encoder.joblib")
    return model, route_encoder, day_encoder


@st.cache_data
def load_routes():
    return pd.read_csv(ROUTES_FILE)


def main():
    st.set_page_config(page_title="Berlin Transit Delay Predictor", page_icon="🚌")
    st.title("🚌 Berlin Transit Delay Predictor")
    st.caption(
        "Trained on real, self-collected data from VBB's live feed — "
        "predicts delay risk for a route, day, and time, before you even leave home."
    )

    model, route_encoder, day_encoder = load_model_and_encoders()
    routes_df = load_routes()

    # Only offer routes the model actually learned about during training.
    known_route_ids = set(route_encoder.classes_)
    routes_df = routes_df[routes_df["route_id"].isin(known_route_ids)]

    col1, col2 = st.columns(2)
    with col1:
        selected_display = st.selectbox(
            "Route", sorted(routes_df["display_name"].unique())
        )
        selected_day = st.selectbox("Day of week", DAYS)
    with col2:
        selected_hour = st.slider("Hour of day (24h)", 0, 23, 8)

    if st.button("Predict delay risk", type="primary"):
        route_id = routes_df.loc[
            routes_df["display_name"] == selected_display, "route_id"
        ].iloc[0]

        route_encoded = route_encoder.transform([route_id])[0]
        day_encoded = day_encoder.transform([selected_day])[0]

        features = pd.DataFrame(
            [[route_encoded, day_encoded, selected_hour]],
            columns=["route_encoded", "day_encoded", "hour"],
        )

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # probability of "Delayed"

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ Likely DELAYED — {probability:.0%} estimated risk")
            st.write("Consider leaving a few minutes earlier for this trip.")
        else:
            st.success(f"✅ Likely ON TIME — {probability:.0%} estimated delay risk")

        st.caption(
            "Based on patterns learned from real historical data. "
            "Not a live status check — see VBB's app for real-time updates."
        )

    with st.expander("How this works"):
        st.write(
            "This isn't checking live GPS data — VBB's own app already does that. "
            "Instead, this model learned patterns from weeks of real delay data "
            "I collected myself, to answer a question live tracking can't: "
            "'is this route generally reliable at this day and time?' — "
            "useful for planning *before* you even leave home."
        )


if __name__ == "__main__":
    main()