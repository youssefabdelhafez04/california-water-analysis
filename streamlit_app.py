"""Streamlit dashboard for the precomputed California HUC4 extracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HUC4_FEATURE_PATH = "properties.huc4"
DEFAULT_SESSION_QUESTION_LIMIT = 10

st.set_page_config(page_title="California Irrigation & Precipitation", layout="wide")


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, dict]:
    data = pd.read_csv(RESULTS / "dashboard_huc4_year.csv", dtype={"huc4": str})
    geometry = json.loads((RESULTS / "dashboard_huc4.geojson").read_text())
    return data, geometry


def secret_or_environment(name: str, default: str | None = None) -> str | None:
    """Read local environment settings or Streamlit Community Cloud secrets."""
    environment_value = os.environ.get(name)
    if environment_value:
        return environment_value
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        value = None
    return str(value) if value is not None else default


def session_question_limit() -> int:
    raw_limit = secret_or_environment(
        "CHAT_QUESTION_LIMIT", str(DEFAULT_SESSION_QUESTION_LIMIT)
    )
    try:
        return max(1, int(raw_limit or DEFAULT_SESSION_QUESTION_LIMIT))
    except ValueError:
        return DEFAULT_SESSION_QUESTION_LIMIT


data, geometry = load_dashboard_data()
st.title("California irrigation use and precipitation")
st.caption("Try the Chat with the data tab to ask questions in plain English.")
st.caption(
    "Precomputed HUC4 summaries from USGS modeled irrigation/effective precipitation and NOAA stations. "
    "The dashboard never loads the raw wide source files."
)

dashboard_tab, chat_tab = st.tabs(["Dashboard", "Chat with the data"])

with dashboard_tab:
    year = st.sidebar.slider("Year", int(data.year.min()), int(data.year.max()), int(data.year.max()))
    watersheds = sorted(data["watershed"].unique())
    selected = st.sidebar.multiselect("Watershed", watersheds, default=watersheds)
    metric_labels = {
        "Irrigation consumptive use (MGD)": "irrigation_mgd",
        "Effective precipitation (MGD equivalent)": "effective_precipitation_mgd",
        "NOAA station precipitation (inches)": "precipitation_in",
    }
    metric_label = st.sidebar.selectbox("Map metric", list(metric_labels))
    metric = metric_labels[metric_label]

    current = data[data["year"].eq(year) & data["watershed"].isin(selected)].copy()
    fig = px.choropleth(
        current,
        geojson=geometry,
        locations="huc4",
        featureidkey=HUC4_FEATURE_PATH,
        color=metric,
        hover_name="watershed",
        hover_data={
            "huc4": True,
            "irrigation_mgd": ":.2f",
            "effective_precipitation_mgd": ":.2f",
            "precipitation_in": ":.2f",
            "temperature_f": ":.1f",
            "huc12_count": True,
        },
        color_continuous_scale="Blues",
        labels={metric: metric_label},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 35, "l": 0, "b": 0}, height=750)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Selected watershed trends")
    top_irrigation_watersheds = (
        data.groupby("watershed", as_index=False)["irrigation_mgd"]
        .sum()
        .nlargest(5, "irrigation_mgd")["watershed"]
        .tolist()
    )
    trend_watersheds = st.multiselect(
        "Trend watersheds",
        watersheds,
        default=top_irrigation_watersheds,
    )
    log_scale = st.checkbox("Log scale", value=False)
    trend = data[data["watershed"].isin(trend_watersheds)].copy()
    trend_fig = px.line(
        trend,
        x="year",
        y=metric,
        color="watershed",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={metric: metric_label, "year": "Year"},
    )
    trend_fig.update_yaxes(type="log" if log_scale else "linear")
    st.plotly_chart(trend_fig, use_container_width=True)

    with st.expander("Data and interpretation notes"):
        st.markdown(
            "HUC4 totals aggregate the available HUC12 model outputs. NOAA precipitation and temperature are "
            "unweighted means from up to three stations per HUC4. Effective precipitation and irrigation "
            "consumptive use are jointly generated USGS model quantities, so displayed relationships are descriptive."
        )
        st.dataframe(current.sort_values("watershed"), use_container_width=True, hide_index=True)

with chat_tab:
    st.subheader("Chat with the data")
    st.caption(
        "Ask a plain-English question. The agent inspects the SQLite schema and runs a read-only SELECT query."
    )
    if "water_chat" not in st.session_state:
        st.session_state.water_chat = []
    if "water_chat_questions_used" not in st.session_state:
        st.session_state.water_chat_questions_used = 0

    question_limit = session_question_limit()
    questions_used = st.session_state.water_chat_questions_used
    questions_remaining = max(0, question_limit - questions_used)

    for message in st.session_state.water_chat:
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            st.write(message["answer"])
            if message.get("sql"):
                st.code(message["sql"], language="sql")
            if message.get("columns"):
                st.dataframe(
                    pd.DataFrame(message["rows"], columns=message["columns"]),
                    use_container_width=True,
                    hide_index=True,
                )

    st.caption(
        f"Questions remaining in this browser session: {questions_remaining}/{question_limit}. "
        "The cap protects the public demo from unbounded API usage."
    )
    if questions_remaining:
        question = st.chat_input("Which watershed used the most irrigation water in 2015?")
    else:
        st.warning("This session has reached its question limit. Start a new session to continue.")
        question = None
    if question:
        # Count submissions, including failed calls, so errors cannot bypass the public limit.
        st.session_state.water_chat_questions_used += 1
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Querying the water database..."):
                try:
                    from water_policy.chat_agent import WaterDataAgent

                    api_key = secret_or_environment("OPENAI_API_KEY")
                    if not api_key:
                        raise RuntimeError(
                            "OPENAI_API_KEY is not configured for this deployment."
                        )
                    os.environ["OPENAI_API_KEY"] = api_key
                    agent = WaterDataAgent(RESULTS / "water_analysis.sqlite")
                    response = agent.ask(question)
                    st.write(response.answer)
                    if response.sql:
                        st.code(response.sql, language="sql")
                    if response.result:
                        st.dataframe(
                            pd.DataFrame(response.result.rows, columns=response.result.columns),
                            use_container_width=True,
                            hide_index=True,
                        )
                    saved = {
                        "question": question,
                        "answer": response.answer,
                        "sql": response.sql,
                        "columns": response.result.columns if response.result else [],
                        "rows": response.result.rows if response.result else [],
                    }
                except Exception as exc:
                    message = f"The data agent is unavailable: {exc}"
                    st.error(message)
                    saved = {"question": question, "answer": message}
            st.session_state.water_chat.append(saved)
