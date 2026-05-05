from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import folium
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from streamlit_folium import st_folium

DEFAULT_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
MODEL_PATH = Path("housing_model.pkl")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Sans+3:wght@400;600&display=swap');

          :root {
            --ink: #132032;
            --teal: #0f766e;
            --mint: #c8f1e2;
            --sand: #fff8ee;
            --sun: #fbbf24;
          }

          .stApp {
            background:
              radial-gradient(circle at 12% 18%, rgba(251, 191, 36, 0.18), transparent 26%),
              radial-gradient(circle at 80% 14%, rgba(15, 118, 110, 0.18), transparent 36%),
              linear-gradient(160deg, var(--sand) 0%, #f8fcff 55%, #f7fffb 100%);
          }

          .block-container {
            padding-top: 1.4rem;
            max-width: 1180px;
            animation: rise-in 450ms ease-out;
          }

          h1, h2, h3, h4 {
            font-family: "Space Grotesk", "Avenir Next", sans-serif !important;
            color: var(--ink);
          }

          p, label, div, .stMarkdown {
            font-family: "Source Sans 3", "Trebuchet MS", sans-serif !important;
          }

          /* Keep Streamlit icon fonts intact (prevents literal text like "arrow_down"). */
          .material-symbols-rounded,
          .material-symbols-outlined,
          [class*="material-symbols"] {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
          }

          /* Force high-contrast text on light background, even with dark Streamlit theme defaults */
          .stApp,
          .stApp p,
          .stApp li,
          .stApp label,
          .stApp div[data-testid="stMarkdownContainer"] *,
          .stApp div[data-testid="stHeadingWithActionElements"] h1,
          .stApp div[data-testid="stHeadingWithActionElements"] h2,
          .stApp div[data-testid="stHeadingWithActionElements"] h3,
          .stApp div[data-testid="stHeadingWithActionElements"] h4 {
            color: var(--ink) !important;
          }

          .stApp div[data-testid="stCaptionContainer"],
          .stApp div[data-testid="stCaptionContainer"] * {
            color: #3b556f !important;
          }

          .stApp div[data-testid="stWidgetLabel"] p {
            color: var(--ink) !important;
            font-weight: 600 !important;
          }

          .stApp input,
          .stApp textarea {
            color: var(--ink) !important;
          }

          /* Number input controls: force light surfaces in dark-theme runtimes */
          .stApp div[data-testid="stNumberInputContainer"] input {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #b8c8d6 !important;
          }

          .stApp div[data-testid="stNumberInputContainer"] button {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #b8c8d6 !important;
          }

          .stApp div[data-testid="stNumberInputContainer"] button:hover {
            background: #f3f8fc !important;
          }

          .hero-card {
            border: 1px solid rgba(15, 118, 110, 0.24);
            border-radius: 18px;
            background: linear-gradient(130deg, rgba(200, 241, 226, 0.75), rgba(255, 248, 238, 0.92));
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
          }

          .hero-card h2,
          .hero-card p {
            color: var(--ink) !important;
          }

          .status-chip {
            display: inline-block;
            background: #e7f8f5;
            color: #0f4f4a;
            border: 1px solid rgba(15, 118, 110, 0.35);
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.8rem;
            margin-right: 0.4rem;
          }

          @keyframes rise-in {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    if "latitude" not in st.session_state:
        st.session_state.latitude = 36.7783
    if "longitude" not in st.session_state:
        st.session_state.longitude = -119.4179
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


@st.cache_resource
def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"source": "none", "model": None, "feature_names": DEFAULT_FEATURES}

    try:
        with path.open("rb") as file:
            obj = pickle.load(file)
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "broken-artifact",
            "model": None,
            "feature_names": DEFAULT_FEATURES,
            "error": str(exc),
        }

    if isinstance(obj, dict) and "model" in obj:
        feature_names = list(obj.get("feature_names", DEFAULT_FEATURES))
        return {
            "source": "artifact",
            "model": obj["model"],
            "feature_names": feature_names,
            "metrics": obj.get("metrics", {}),
        }

    if hasattr(obj, "predict"):
        return {"source": "artifact", "model": obj, "feature_names": DEFAULT_FEATURES}

    return {"source": "unknown-artifact", "model": None, "feature_names": DEFAULT_FEATURES}


@st.cache_resource
def build_fallback_model() -> dict[str, Any]:
    try:
        data = fetch_california_housing(as_frame=True)
        X = data.data[DEFAULT_FEATURES]
        y = data.target

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return {"source": "fallback-trained", "model": model, "feature_names": DEFAULT_FEATURES}
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "heuristic-demo",
            "model": None,
            "feature_names": DEFAULT_FEATURES,
            "error": str(exc),
        }


def heuristic_prediction(features: dict[str, float]) -> float:
    median_income = features["MedInc"] * 68000
    room_effect = (features["AveRooms"] - 5.0) * 17000
    bedroom_penalty = max(0.0, features["AveBedrms"] - 1.2) * 36000
    age_effect = max(0.0, 1 - abs(features["HouseAge"] - 25) / 75) * 28000
    population_effect = np.log1p(features["Population"]) * 2600
    occupancy_penalty = max(0.0, features["AveOccup"] - 3.0) * 9000
    coast_bonus = 50000 if features["Longitude"] > -122 and features["Latitude"] < 38 else 0
    raw_price = (
        median_income
        + room_effect
        - bedroom_penalty
        + age_effect
        + population_effect
        - occupancy_penalty
        + coast_bonus
    )
    clipped = float(np.clip(raw_price, 70000, 1_250_000))
    return clipped / 100000


def make_prediction(engine: dict[str, Any], features: dict[str, float]) -> float:
    model = engine.get("model")
    if model is None:
        return heuristic_prediction(features)

    ordered_features = [features[name] for name in engine["feature_names"]]
    pred = model.predict(np.array(ordered_features).reshape(1, -1))
    return float(pred[0])


def map_widget() -> None:
    st.subheader("Pick Location")
    st.caption("Click anywhere in California to set latitude and longitude.")
    ca_map = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude],
        zoom_start=6,
        tiles="CartoDB positron",
    )
    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        tooltip="Selected point",
        icon=folium.Icon(color="green", icon="home"),
    ).add_to(ca_map)
    result = st_folium(ca_map, height=420, width=None, returned_objects=["last_clicked"])

    click = result.get("last_clicked") if result else None
    if click:
        new_lat = float(click["lat"])
        new_lon = float(click["lng"])
        if 32.0 <= new_lat <= 42.5 and -124.8 <= new_lon <= -114.0:
            if (abs(new_lat - st.session_state.latitude) > 1e-8) or (
                abs(new_lon - st.session_state.longitude) > 1e-8
            ):
                st.session_state.latitude = new_lat
                st.session_state.longitude = new_lon
                st.rerun()
        else:
            st.info("Map click outside California bounds. Keep your point inside the state range.")


def input_panel() -> dict[str, float]:
    st.subheader("Home + Neighborhood Inputs")
    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.slider("Median income (x$10,000)", 0.5, 15.0, 4.2, 0.1)
        house_age = st.slider("House age (years)", 1, 52, 24, 1)
        avg_rooms = st.slider("Average rooms", 1.0, 15.0, 5.4, 0.1)
        avg_bedrms = st.slider("Average bedrooms", 0.5, 5.0, 1.1, 0.1)

    with col2:
        population = st.slider("Population", 100, 20000, 2800, 100)
        avg_occup = st.slider("Average occupancy", 1.0, 8.0, 2.8, 0.1)
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.5,
            value=float(st.session_state.latitude),
            step=0.0001,
            format="%.4f",
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-124.8,
            max_value=-114.0,
            value=float(st.session_state.longitude),
            step=0.0001,
            format="%.4f",
        )

    st.session_state.latitude = latitude
    st.session_state.longitude = longitude
    return {
        "MedInc": med_inc,
        "HouseAge": float(house_age),
        "AveRooms": avg_rooms,
        "AveBedrms": avg_bedrms,
        "Population": float(population),
        "AveOccup": avg_occup,
        "Latitude": float(latitude),
        "Longitude": float(longitude),
    }


def results_panel(price_units: float) -> None:
    predicted_usd = price_units * 100000
    st.subheader("Prediction")
    st.metric("Estimated Median Home Value", f"${predicted_usd:,.0f}")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted_usd,
            number={"prefix": "$", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [70000, 1250000]},
                "bar": {"color": "#0f766e"},
                "steps": [
                    {"range": [70000, 300000], "color": "#effbf6"},
                    {"range": [300000, 700000], "color": "#d8f4eb"},
                    {"range": [700000, 1250000], "color": "#c8f1e2"},
                ],
            },
            title={"text": "Value Range Position"},
        )
    )
    gauge.update_layout(height=290, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(gauge, use_container_width=True)


def history_panel() -> None:
    history = st.session_state.prediction_history
    if len(history) < 2:
        st.caption("Run multiple predictions to see trend history.")
        return

    st.subheader("Session Prediction History")
    x = list(range(1, len(history) + 1))
    y = [row["predicted_usd"] for row in history]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line={"color": "#0f766e", "width": 3},
                marker={"size": 8, "color": "#f59e0b"},
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Prediction Run",
        yaxis_title="Predicted Median Value (USD)",
        height=280,
        margin=dict(l=20, r=10, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="CAmarket Housing Estimator", layout="wide")
    inject_styles()
    initialize_state()

    artifact_engine = load_artifact(MODEL_PATH)
    engine = artifact_engine
    if engine.get("model") is None:
        engine = build_fallback_model()

    st.markdown(
        """
        <div class="hero-card">
          <h2>CAmarket: California Housing Value Estimator</h2>
          <p>Estimate California median home value from location and neighborhood inputs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    engine_label = {
        "artifact": "Model: Project Trained",
        "fallback-trained": "Model: Baseline Regression",
        "heuristic-demo": "Model: Demo Estimator",
        "none": "Model: Baseline Regression",
        "broken-artifact": "Model: Baseline Regression",
        "unknown-artifact": "Model: Baseline Regression",
    }.get(engine.get("source", "none"), "Model: Baseline Regression")

    st.markdown(f'<span class="status-chip">{engine_label}</span>', unsafe_allow_html=True)
    if engine.get("error"):
        st.caption("Using baseline model in this session.")

    col_left, col_right = st.columns([1.15, 1.0], gap="large")
    with col_left:
        map_widget()
    with col_right:
        features = input_panel()
        if st.button("Predict Median Value", use_container_width=True, type="primary"):
            pred_units = make_prediction(engine, features)
            st.session_state.prediction_history.append(
                {"predicted_usd": pred_units * 100000, "features": features}
            )
            results_panel(pred_units)
        elif st.session_state.prediction_history:
            last = st.session_state.prediction_history[-1]["predicted_usd"] / 100000
            results_panel(last)
        else:
            st.info("Press **Predict Median Value** to generate an estimate.")

    history_panel()

if __name__ == "__main__":
    main()
