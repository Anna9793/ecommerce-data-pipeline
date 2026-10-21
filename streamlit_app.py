import os
import streamlit as st
import requests
import pandas as pd
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

# Sidebar Ingestion Simulator & Multi-Tenant Controls
with st.sidebar:
    st.title("🏪 Multi-Tenant Platform")
    active_store = st.selectbox(
        "Active Store / Tenant",
        ["giftshop_uk", "nordic_tech"],
        format_func=lambda x: {
            "giftshop_uk": "🎁 GiftShop UK (Vintage Retail)",
            "nordic_tech": "🎧 NordicWear & Tech (Shopify)"
        }.get(x, x)
    )
    
    if active_store == "nordic_tech":
        st.info("🔹 **Adapter**: `ShopifyAdapter`\n🔹 **Format**: Shopify Order Export\n🔹 **Catalog**: Smart Audio & Activewear")
    else:
        st.info("🔹 **Adapter**: `UciRetailAdapter`\n🔹 **Format**: UCI Online Retail\n🔹 **Catalog**: Vintage Home & Gifts")
        
    st.divider()
    st.title("🔴 Live Ingestion Simulator")
    st.write("Stream mock purchases/cancellations directly to BigQuery to simulate live retail changes.")
    
    sim_mode = st.selectbox(
        "Simulation Mode",
        ["standard", "drift_cancellations", "drift_velocity"],
        format_func=lambda x: {
            "standard": "Standard Buying (Normal)",
            "drift_cancellations": "Spike Cancellations (Drift)",
            "drift_velocity": "Spike Order Size (Drift)"
        }.get(x, x)
    )
    
    num_records = st.slider("Records to Generate", min_value=10, max_value=200, value=50, step=10)
    
    if st.button("⚡ Stream Simulated Transactions", use_container_width=True):
        with st.spinner("Streaming transactions to BigQuery..."):
            try:
                resp = requests.post(f"{API_URL}/simulate?mode={sim_mode}&num_records={num_records}")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "success":
                        st.success(f"✅ {data.get('message')}")
                        # Clear Streamlit's cached data so the new transactions appear instantly
                        st.cache_data.clear()
                    else:
                        st.error("❌ Simulation Failed inside API Service:")
                        st.error(data.get("message"))
                        with st.expander("View Detailed Exception Traceback"):
                            st.code(data.get("traceback"))
                else:
                    st.error(f"❌ Gateway returned status code {resp.status_code}")
                    st.code(resp.text[:800])
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

st.title("🛍️ E-commerce Analytics & ML Dashboard")
st.caption("Real-time Customer Segmentation & Churn Risk Analytics")

# Define Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Customer Segmentation", 
    "🔮 Churn Prediction", 
    "🛍️ Product Advisor (RAG & pgvector)",
    "📈 Model Monitoring"
])

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

        st.subheader("⚙️ Model Operations")
        with st.container(border=True):
            st.write("Trigger serverless model retraining in the cloud on GCP Vertex AI using your transactions data.")
            if st.button("⚡ Trigger Cloud Churn Retraining", use_container_width=True):
                with st.spinner("Submitting training job to Vertex AI..."):
                    try:
                        resp = requests.post(f"{API_URL}/train/churn")
                        if resp.status_code == 200:
                            data = resp.json()
                            console_url = data.get("console_url")
                            if console_url:
                                st.success("✅ Retraining pipeline submitted successfully!")
                                st.markdown(f"[🔗 **Click here to monitor the pipeline run in Vertex AI Console**]({console_url})")
                            else:
                                st.success("✅ Retraining job submitted successfully!")
                        else:
                            st.error(f"❌ Failed to submit job: {resp.text}")
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")

# TAB 4: MODEL MONITORING & DRIFT
with tab4:
    st.subheader("📈 Real-time Feature Drift Monitoring")
    st.write("This panel tracks statistical drift between the original training baseline and live streamed customer metrics in BigQuery using the Kolmogorov-Smirnov (K-S) test.")

    # Trigger GET /monitoring/drift
    try:
        drift_resp = requests.get(f"{API_URL}/monitoring/drift")
        if drift_resp.status_code == 200:
            drift_data = drift_resp.json()
            
            # 1. Overall Health Badge
            drift_detected = drift_data.get("drift_detected", False)
            if drift_detected:
                st.error("🔴 **DATA DRIFT DETECTED: Retraining Required!**")
                
                # Check-and-retrain button
                col_btn, col_info = st.columns([1, 2])
                with col_btn:
                    if st.button("⚡ Run Closed-Loop Check & Retrain", use_container_width=True):
                        with st.spinner("Checking drift and triggering Vertex AI Pipeline..."):
                            try:
                                retrain_resp = requests.post(f"{API_URL}/monitoring/check-and-retrain")
                                if retrain_resp.status_code == 200:
                                    retrain_data = retrain_resp.json()
                                    if retrain_data.get("status") == "drift_detected":
                                        st.success("🚀 Drift confirmed! Retraining pipeline submitted successfully!")
                                        console_url = retrain_data.get("console_url")
                                        if console_url:
                                            st.markdown(f"[🔗 **Monitor Retraining Job in Vertex Console**]({console_url})")
                                    else:
                                        st.info("ℹ️ " + retrain_data.get("message"))
                                else:
                                    st.error(f"❌ Webhook call failed: {retrain_resp.text}")
                            except Exception as e:
                                st.error(f"❌ Connection error: {str(e)}")
            else:
                st.success("🟢 **SYSTEM STATUS HEALTHY: No Significant Drift Detected**")

            st.divider()

            # 2. Features metrics table
            st.subheader("📊 Feature Drift Summary Table")
            features_dict = drift_data.get("features", {})
            
            if features_dict:
                table_rows = []
                for feat, stats in features_dict.items():
                    table_rows.append({
                        "Feature": feat,
                        "K-S Statistic": round(stats["ks_statistic"], 4),
                        "p-value": f"{stats['p_value']:.4e}" if stats['p_value'] < 0.0001 else round(stats['p_value'], 4),
                        "Baseline Mean": round(stats["baseline_mean"], 2),
                        "Target Mean": round(stats["target_mean"], 2),
                        "Status": "🔴 DRIFTED" if stats["drifted"] else "🟢 Healthy"
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                st.divider()

                # 3. Distribution chart selection
                st.subheader("📊 Distribution Overlay Plot")
                selected_feat = st.selectbox("Select a feature to view distribution overlay", list(features_dict.keys()))
                
                feat_stats = features_dict[selected_feat]
                baseline_vals = feat_stats.get("baseline_values", [])
                target_vals = feat_stats.get("target_values", [])
                
                if baseline_vals and target_vals:
                    # Create overlay histogram using plotly
                    plot_df = pd.concat([
                        pd.DataFrame({"value": baseline_vals, "Dataset": "Baseline (Training)"}),
                        pd.DataFrame({"value": target_vals, "Dataset": "Live Stream (Target)"})
                    ])
                    
                    fig = px.histogram(
                        plot_df, 
                        x="value", 
                        color="Dataset", 
                        barmode="overlay",
                        title=f"Distribution Comparison: {selected_feat}",
                        labels={"value": selected_feat},
                        histnorm="probability density",
                        marginal="box",
                        opacity=0.6,
                        color_discrete_map={"Baseline (Training)": "#1f77b4", "Live Stream (Target)": "#ff7f0e"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No feature metrics returned in the drift report.")
        else:
            st.error(f"Failed to fetch drift report: status {drift_resp.status_code}")
            st.code(drift_resp.text)
    except Exception as e:
        st.error(f"Error fetching drift report from API: {str(e)}")

    st.divider()

    # BigQuery Time Travel Explorer
    with st.expander("🕰️ **BigQuery Time Travel Explorer**", expanded=False):
        st.markdown("""
        **Point-in-Time Reproducibility & Auditing**:
        Explore BigQuery `FOR SYSTEM_TIME AS OF` time-travel queries for reproducible auditing and rollback.
        """)

        st.markdown("#### 📜 BigQuery Time Travel Query Builder")
        time_interval = st.selectbox(
            "Select Point-in-Time Snapshot Window",
            ["1 Hour Ago (Interval 1 HOUR)", "1 Day Ago (Interval 1 DAY)", "7 Days Ago (Interval 7 DAY)", "Custom UTC Timestamp"]
        )
        
        if "1 Hour" in time_interval:
            time_expr = "TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)"
        elif "1 Day" in time_interval:
            time_expr = "TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)"
        elif "7 Days" in time_interval:
            time_expr = "TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)"
        else:
            time_expr = "TIMESTAMP '2026-09-01 00:00:00 UTC'"

        sql_sample = f"""-- BigQuery Point-in-Time Time Travel Query (Zero Compute Overhead)
SELECT 
    customer_id, 
    COUNT(DISTINCT invoice_no) AS total_orders, 
    ROUND(SUM(quantity * unit_price), 2) AS historical_spend
FROM `anna-ml-pipeline.retail_data.transactions`
FOR SYSTEM_TIME AS OF {time_expr}
WHERE customer_id IS NOT NULL
GROUP BY customer_id
LIMIT 10;"""
        st.code(sql_sample, language="sql")
        st.caption("⚡ Uses BigQuery's native 7-day immutable time-travel window for instant rollback & reproducibility.")


# TAB 3: PRODUCT ADVISOR CHATBOT (RAG + PGVECTOR)
with tab3:
    store_title = "NordicWear & Tech (Shopify)" if active_store == "nordic_tech" else "GiftShop UK (Vintage Retail)"
    st.subheader(f"🛍️ Product Advisor Chatbot — {store_title}")
    st.write(f"Ask natural questions about **{store_title}**. The assistant searches catalog vector embeddings with real-time budget filtering and uses **Gemini** for personalized recommendations.")

    # Dynamic Store Departments Directory
    with st.expander(f"🏪 **Explore {store_title} Departments & Catalog Specialties**", expanded=True):
        if active_store == "nordic_tech":
            st.markdown("""
            Our Scandinavian catalog features high-performance activewear, audio, and workspace technology:
            * 🎧 **Smart Audio & Acoustics**: Nordic Pro ANC Wireless Headphones, Studio Monitor Earbuds.
            * 🏃 **Performance Activewear**: Merino Wool Thermal Hoodies, Trail Running Caps, Anti-Blister Compression Socks.
            * 🏔️ **All-Weather Mountain Apparel**: Gore-Tex All-Weather Mountain Parkas, Technical Outerwear.
            * ⌚ **Smart Wearables & Gadgets**: UltraLight Titanium Smartwatches, Smart LED Temperature Water Bottles.
            * ⌨️ **Ergonomic Workspace & Gaming**: Mechanical RGB Hot-Swap Keyboards, Ergonomic Mice, Minimalist Aluminum Laptop Stands.
            """)
        else:
            st.markdown("""
            Our boutique catalog features curated lifestyle, home styling, and gift collections:
            * 🏠 **Home Decor & Lighting**: Hanging lanterns, romantic T-light holders, wall clocks, ambient candles, decorative mirrors.
            * ☕ **Kitchen & Dining**: Regency 3-tier cakestands, vintage tea sets, ceramic mugs, cutlery, retro dining accessories.
            * 🎄 **Holiday & Seasonal**: Handcrafted Christmas decorations, paper chain kits, winter festive gifts.
            * 👝 **Storage & Accessories**: Vintage tote bags, trinket tins, vanity storage, stylish cases.
            * 🎉 **Party & Celebration**: Festive bunting, party garlands, celebration accessories, greeting supplies.
            * 🧸 **Kids & Novelty Toys**: Playful gifts, wooden toys, vintage pencil sets, children's novelties.
            """)

    # Top search controls
    col_filters, col_presets = st.columns([1, 2])
    with col_filters:
        with st.container(border=True):
            st.markdown("#### 🎯 Search Constraints")
            use_budget = st.toggle("Apply Maximum Budget", value=True)
            budget_max = None
            if use_budget:
                max_slider_val = 250.0 if active_store == "nordic_tech" else 100.0
                default_budget = 150.0 if active_store == "nordic_tech" else 25.0
                budget_max = st.slider("Max Budget ($)", min_value=5.0, max_value=max_slider_val, value=default_budget, step=5.0)
            top_k_val = st.selectbox("Products to Retrieve", [2, 3, 4, 6], index=2)

    with col_presets:
        with st.container(border=True):
            st.markdown("#### 💡 Quick Search Ideas")
            col_b1, col_b2, col_b3 = st.columns(3)
            quick_query = None
            if active_store == "nordic_tech":
                if col_b1.button("🎧 ANC Headphones", use_container_width=True):
                    quick_query = "I need high-performance noise cancelling wireless headphones for travel under $150."
                if col_b2.button("🧥 Thermal Winter Hoodie", use_container_width=True):
                    quick_query = "Show me merino wool thermal hoodies or mountain jackets for cold outdoor weather."
                if col_b3.button("⌨️ Ergonomic Desk Gear", use_container_width=True):
                    quick_query = "What do you have for ergonomic workspace setups, mechanical keyboards, and laptop stands?"
            else:
                if col_b1.button("🎄 Cozy Winter Gifts", use_container_width=True):
                    quick_query = "I need a warm and cozy holiday gift for a winter evening under $25."
                if col_b2.button("☕ Vintage Kitchenware", use_container_width=True):
                    quick_query = "Show me vintage retro kitchenware, tea sets, and cute dining accessories."
                if col_b3.button("🎉 Party & Bunting", use_container_width=True):
                    quick_query = "What do you have for party decorations, garlands, and festive celebrations?"

    # Initialize chat history
    if "advisor_chat_history" not in st.session_state:
        st.session_state["advisor_chat_history"] = []

    # Display past messages
    for msg in st.session_state["advisor_chat_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                data = msg["content"]
                st.write(data["intro_message"])
                
                # Render product recommendation cards in 2-column grid
                recs = data.get("recommendations", [])
                if recs:
                    cols = st.columns(min(len(recs), 2))
                    for idx, p in enumerate(recs):
                        col_target = cols[idx % len(cols)]
                        with col_target:
                            with st.container(border=True):
                                st.markdown(f"### 🏷️ {p['description']}")
                                st.caption(f"**Category:** {p.get('category', 'General')} | **Code:** `{p['stock_code']}`")
                                st.metric("Price", f"${p['unit_price']:.2f}", delta=f"{p.get('similarity', 0.85)*100:.1f}% Match")
                                st.info(f"💡 **Why Recommended:** {p.get('why_recommended', 'Great match for your query.')}")
                                
                if data.get("shopping_tip"):
                    st.success(f"✨ **Shopping Tip:** {data['shopping_tip']}")

    # User chat input
    user_input = st.chat_input(f"Ask for product advice at {store_title}...")
    active_prompt = quick_query or user_input

    if active_prompt:
        # Append and display user message
        st.session_state["advisor_chat_history"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.write(active_prompt)

        # Query RAG Advisor Endpoint with active tenant
        with st.chat_message("assistant"):
            with st.spinner(f"Searching {store_title} catalog and generating recommendations with Gemini..."):
                try:
                    payload = {
                        "query": active_prompt,
                        "budget_max": float(budget_max) if budget_max else None,
                        "top_k": int(top_k_val),
                        "tenant_id": active_store
                    }
                    resp = requests.post(f"{API_URL}/rag/advisor", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["advisor_chat_history"].append({"role": "assistant", "content": data})
                        
                        st.write(data["intro_message"])
                        
                        recs = data.get("recommendations", [])
                        if recs:
                            cols = st.columns(min(len(recs), 2))
                            for idx, p in enumerate(recs):
                                col_target = cols[idx % len(cols)]
                                with col_target:
                                    with st.container(border=True):
                                        st.markdown(f"### 🏷️ {p['description']}")
                                        st.caption(f"**Category:** {p.get('category', 'General')} | **Code:** `{p['stock_code']}`")
                                        st.metric("Price", f"${p['unit_price']:.2f}", delta=f"{p.get('similarity', 0.85)*100:.1f}% Match")
                                        st.info(f"💡 **Why Recommended:** {p.get('why_recommended', 'Great match for your query.')}")
                                        
                        if data.get("shopping_tip"):
                            st.success(f"✨ **Shopping Tip:** {data['shopping_tip']}")
                    else:
                        st.error(f"Failed to fetch product advice: {resp.status_code}")
                        st.code(resp.text)
                except Exception as e:
                    st.error(f"Error communicating with RAG Advisor API: {str(e)}")