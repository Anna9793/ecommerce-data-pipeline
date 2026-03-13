import streamlit as st
import pandas as pd 
from pathlib import Path

st.title("Customer Segmentation Dashboard")

profile = pd.read_csv("data/processed/cluster_profile.csv")

st.header("Segment Summary")

st.dataframe(profile)

clusters = pd.read_csv("data/processed/rfm_train_clusters.csv")

segment_counts = clusters["segment"].value_counts()

st.subheader ("Customer Distribtion by Segment")

st.bar_chart(segment_counts)

segment = st.selectbox(
    "Select segment",
    clusters["segment"].unique().tolist()
    )

filtered = clusters[clusters["segment"] == segment]

st.write(f"Customers in segment :{segment}")
st.dataframe(filtered.head(20))

st.image("reports/cluster_visualization.png")