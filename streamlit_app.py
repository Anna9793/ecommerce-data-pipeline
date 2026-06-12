import streamlit as st
import requests
import plotly.express as px

from app.service import MODEL_VERSION
from app.repository import (get_total_predictions,
                           get_average_response_time,
                           get_latest_predictions,
                           get_segment_distribution,
                           get_predictions_by_model_version)

if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

st.title("Customer Segmentation Dashboard")

count = get_total_predictions()

avg_response = get_average_response_time()

@st.cache_data
def load_latest_predictions():
    return get_latest_predictions()

latest_df = load_latest_predictions()

@st.cache_data
def load_segment_distribution():
    return get_segment_distribution()

segment_df = load_segment_distribution()

version_df = get_predictions_by_model_version()

with st.container():
    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Predictions",
            value=count
        )

    with col2:
        st.metric(
            label="Average Response Time(ms)",
            value=round(avg_response,2)
        )

    with col3:
        st.metric(
            label="Production Model",
            value=MODEL_VERSION
        )

st.divider()

with st.container():
    st.subheader("Latest Predictions")

    selected_segment = st.selectbox(
        "Show segment",
        ["All"] + sorted(latest_df["label"].unique())
    )

    if selected_segment != "All":
        latest_df = latest_df[
            latest_df["label"] == selected_segment
        ]

    st.dataframe(
        latest_df[
            [
                "recency",
                "frequency",
                "avg_order_value",
                "cluster",
                "label"
            ]
        ]
    )

st.divider()

with st.container():
    st.subheader("Segment Distribution")

    st.dataframe(segment_df)

    st.bar_chart(
        segment_df.set_index("label")
    )

st.divider()

with st.container(border=True):
    st.header("Prediction Playground")

    recency = st.number_input(
        "Recency (days)",
        min_value=1.0,
        value=30.0
    )

    frequency = st.number_input(
        "Frequency",
        min_value=1.0,
        value = 5.0
    )

    avg_order_value = st.number_input(
        "Average Order Value",
        min_value=1.0,
        value=100.0
    )

    if st.button("Predict Customer Segment"):
        payload = {
            "recency": recency,
            "frequency":frequency,
            "avg_order_value": avg_order_value
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        if response.status_code == 200:

            result = response.json()

            st.session_state["last_prediction"] = result

            st.cache_data.clear()

            st.rerun()

    if st.session_state["last_prediction"] is not None:

        result = st.session_state["last_prediction"]

        st.success(
            f"Segment: {result['label']}"
        )

        st.write(
            f"Cluster: {result['cluster']}"
        )

st.divider()

with st.container():
    st.subheader("Prediction by Model Version")
    
    fig = px.bar(
        version_df,
        x="predictions",
        y="model_version",
        orientation="h",
    )

    st.plotly_chart(fig)

    