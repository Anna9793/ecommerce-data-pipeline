import os
import streamlit as st
import requests
import plotly.express as px

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

from app.service import MODEL_VERSION, CHURN_MODEL_VERSION
from app.repository import (
    get_total_predictions,
    get_average_response_time,
    get_latest_predictions,
    get_segment_distribution,
    get_predictions_by_model_version,
    get_total_churn_predictions,
    get_average_churn_probability,
    get_latest_churn_predictions
)

# Initialize Session States
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None

if "last_churn_prediction" not in st.session_state:
    st.session_state["last_churn_prediction"] = None

# Title
st.set_page_config(page_title="E-commerce ML Dashboard", layout="wide")
st.title("🛍️ E-commerce Analytics & ML Dashboard")
st.caption("Real-time Customer Segmentation & Churn Risk Analytics")

# Define Tabs
tab1, tab2, tab3 = st.tabs(["📊 Customer Segmentation", "🔮 Churn Prediction", "🤖 AI Marketing Copilot"])

# TAB 1: CUSTOMER SEGMENTATION
with tab1:
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

    st.subheader("System Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Segment Predictions", value=count)
    with col2:
        st.metric(label="Average Response Time (ms)", value=round(avg_response, 2) if avg_response else 0)
    with col3:
        st.metric(label="Segmentation Model version", value=MODEL_VERSION)

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📋 Latest Predictions Logs")
        if not latest_df.empty:
            selected_segment = st.selectbox(
                "Filter by segment",
                ["All"] + sorted(latest_df["label"].dropna().unique())
            )
            filtered_df = latest_df
            if selected_segment != "All":
                filtered_df = latest_df[latest_df["label"] == selected_segment]

            st.dataframe(
                filtered_df[[
                    "recency",
                    "frequency",
                    "avg_order_value",
                    "cluster",
                    "label",
                    "created_at"
                ]],
                use_container_width=True
            )
        else:
            st.info("No predictions found in the database.")

    with col_right:
        st.subheader("🎯 Prediction Playground")
        with st.container(border=True):
            recency = st.number_input("Recency (days)", min_value=0.0, value=30.0, key="seg_recency")
            frequency = st.number_input("Frequency (purchases)", min_value=1.0, value=5.0, key="seg_frequency")
            avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=100.0, key="seg_aov")

            if st.button("Predict Segment", use_container_width=True):
                payload = {
                    "recency": recency,
                    "frequency": frequency,
                    "avg_order_value": avg_order_value
                }
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    st.session_state["last_prediction"] = response.json()
                    st.cache_data.clear()
                    st.rerun()

            if st.session_state["last_prediction"] is not None:
                result = st.session_state["last_prediction"]
                st.success(f"**Segment:** {result['label']}")
                st.info(f"**Cluster ID:** {result['cluster']}")

    st.divider()

    st.subheader("📈 Segment Distribution & Models")
    col_dist, col_ver = st.columns(2)
    with col_dist:
        st.write("Distribution of customers across segments:")
        st.dataframe(segment_df, use_container_width=True)
        if not segment_df.empty:
            fig_pie = px.pie(segment_df, names="label", values="count", title="Segment Share")
            st.plotly_chart(fig_pie, use_container_width=True)
    with col_ver:
        st.write("Predictions count grouped by model version:")
        if not version_df.empty:
            fig_bar = px.bar(version_df, x="predictions", y="model_version", orientation="h", title="Predictions by Model Version")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No model version statistics available.")


# TAB 2: CHURN PREDICTION
with tab2:
    churn_count = get_total_churn_predictions()
    avg_churn_prob = get_average_churn_probability()
    latest_churn_df = get_latest_churn_predictions()

    st.subheader("Churn Model Overview")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.metric(label="Total Churn Predictions", value=churn_count)
    with cc2:
        st.metric(label="Average Churn Risk", value=f"{round(avg_churn_prob * 100, 2)}%")
    with cc3:
        st.metric(label="Churn Model version (Random Forest)", value=CHURN_MODEL_VERSION)

    st.divider()

    col_churn_left, col_churn_right = st.columns([2, 1])

    with col_churn_left:
        st.subheader("📋 Latest Churn Predictions Logs")
        if not latest_churn_df.empty:
            st.dataframe(
                latest_churn_df[[
                    "recency",
                    "frequency",
                    "avg_order_value",
                    "spending_velocity",
                    "cancellation_rate",
                    "preferred_shopping_hour",
                    "churn_probability",
                    "is_churn",
                    "created_at"
                ]],
                use_container_width=True
            )
        else:
            st.info("No churn predictions found in the database.")

    with col_churn_right:
        st.subheader("🔮 Churn Risk Playground")
        with st.container(border=True):
            churn_recency = st.number_input("Recency (days since last purchase)", min_value=0.0, value=90.0, key="churn_rec")
            churn_frequency = st.number_input("Frequency (total purchases)", min_value=1.0, value=2.0, key="churn_freq")
            churn_aov = st.number_input("Average Order Value ($)", min_value=0.0, value=45.0, key="churn_val")
            churn_velocity = st.number_input("Spending Velocity (Ratio 30d/90d)", min_value=0.0, value=1.0, key="churn_vel")
            churn_canc = st.number_input("Cancellation Rate (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="churn_canc")
            churn_hour = st.number_input("Preferred Shopping Hour (0-23)", min_value=0, max_value=23, value=12, key="churn_hour")

            if st.button("Predict Churn Risk", use_container_width=True):
                payload = {
                    "recency": churn_recency,
                    "frequency": churn_frequency,
                    "avg_order_value": churn_aov,
                    "spending_velocity": churn_velocity,
                    "cancellation_rate": churn_canc,
                    "preferred_shopping_hour": int(churn_hour)
                }
                response = requests.post(f"{API_URL}/predict/churn", json=payload)
                if response.status_code == 200:
                    st.session_state["last_churn_prediction"] = response.json()
                    st.cache_data.clear()
                    st.rerun()

            if st.session_state["last_churn_prediction"] is not None:
                churn_res = st.session_state["last_churn_prediction"]
                risk_pct = round(churn_res['churn_probability'] * 100, 1)
                
                if churn_res['is_churn'] == 1:
                    st.error(f"⚠️ **High Churn Risk!** Risk: {risk_pct}%")
                else:
                    st.success(f"✅ **Low Churn Risk** Risk: {risk_pct}%")

# TAB 3: AI MARKETING COPILOT
with tab3:
    st.subheader("🤖 AI Marketing Campaign Generator & Vector Search")
    st.write("Target inactive or high-value customers with personalized recommendations and email drafts powered by Vertex AI and Gemini.")
    
    # Selection
    col_input, col_info = st.columns([1, 2])
    with col_input:
        with st.container(border=True):
            customer_id_input = st.text_input("Enter Customer ID", value="17850")
            st.caption("Try sample customer IDs: **17850**, **13047**, **12583**, **12431**, **14606**")
            generate_btn = st.button("Generate Campaign", use_container_width=True)
            
    with col_info:
        st.info("💡 **How it works:** This assistant identifies the customer's churn risk and segment from BigQuery. It then performs a **Vector Search** to find products similar to their last purchased item, and prompts Gemini to write a personalized email campaign.")

    if generate_btn:
        with st.spinner("Analyzing profile, executing vector similarity search, and generating email campaign..."):
            try:
                response = requests.get(f"{API_URL}/predict/campaign/{customer_id_input}")
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    col_results_left, col_results_right = st.columns([1, 1.5])
                    
                    with col_results_left:
                        st.subheader("👤 Customer Profile Details")
                        profile = result["profile"]
                        
                        # Profile Cards
                        with st.container(border=True):
                            st.write(f"**Customer ID:** {result['customer_id']}")
                            st.write(f"**Segment:** {profile['segment']}")
                            st.write(f"**Last Purchased:** {profile['last_purchased']}")
                            
                            st.write("**Customer Metrics:**")
                            st.write(f"- Recency: {profile['recency']} days")
                            st.write(f"- Frequency: {profile['frequency']} purchases")
                            st.write(f"- Average Order Value: ${profile['avg_order_value']:.2f}")
                            
                            # Churn alert
                            churn_prob = profile.get("churn_probability", 0)
                            is_churn = profile.get("is_churn", 0)
                            if is_churn == 1:
                                st.warning(f"⚠️ Churn Risk: **High ({churn_prob*100:.1f}%)**")
                            else:
                                st.success(f"✅ Churn Risk: **Low ({churn_prob*100:.1f}%)**")
                                
                        st.subheader("🛍️ Vector Search Recommendations")
                        st.write("Top similar items from catalog:")
                        for rec in result["recommendations"]:
                            with st.container(border=True):
                                col_desc, col_sim = st.columns([3, 1])
                                with col_desc:
                                    st.markdown(f"**{rec['description']}**")
                                    st.write(f"Price: ${rec['unit_price']:.2f} | Code: {rec['stock_code']}")
                                with col_sim:
                                    st.metric("Similarity", f"{rec['similarity']*100:.1f}%")
                                    
                    with col_results_right:
                        st.subheader("📧 AI Campaign Copy Draft")
                        with st.container(border=True):
                            st.markdown(result["campaign_draft"])
                            
                else:
                    st.error(f"Failed to generate campaign. Server returned status code: {response.status_code}")
                    st.error(response.text)
            except Exception as e:
                st.error(f"Error connecting to FastAPI API: {str(e)}")