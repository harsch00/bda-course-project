from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import FEATURE_SPEC, DATA_PATH, engineer_features, load_raw_data
from src.inference import build_cost_lookup, evaluate_scenario, load_model_bundle, recommend_margin_and_price


st.set_page_config(page_title="Ecommerce Seller Intelligence", page_icon=":bar_chart:", layout="wide")

APPLE_STYLE = """
<style>
    .stApp { background: #f5f5f7; color: #1d1d1f; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1360px; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e6eb;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }
    h1, h2, h3 { letter-spacing: -0.01em; font-weight: 700; color: #111111; }
    .subtitle { color: #6e6e73; margin-top: -8px; margin-bottom: 14px; }
    .card {
        background: white;
        border: 1px solid #e8e8ed;
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
</style>
"""
st.markdown(APPLE_STYLE, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return engineer_features(load_raw_data(DATA_PATH))


@st.cache_data(show_spinner=False)
def get_momentum(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.assign(Month=df["OrderDate"].dt.to_period("M").dt.to_timestamp())
        .groupby(["Month", "ProductCategory", "ProductName"], as_index=False)["OrderQuantity"]
        .sum()
        .sort_values(["ProductCategory", "ProductName", "Month"])
    )
    monthly["prev_qty"] = monthly.groupby(["ProductCategory", "ProductName"])["OrderQuantity"].shift(1)
    monthly["growth_rate"] = np.where(
        monthly["prev_qty"] > 0,
        (monthly["OrderQuantity"] - monthly["prev_qty"]) / monthly["prev_qty"],
        np.nan,
    )
    monthly["trend"] = pd.cut(
        monthly["growth_rate"], bins=[-np.inf, -0.08, 0.08, np.inf], labels=["Declining", "Stable", "Rising"]
    ).astype(str)
    return monthly


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_model_bundle()


data = get_dataset()
cost_lookup = build_cost_lookup(data)
momentum = get_momentum(data)

st.title("Seller Intelligence Dashboard")
st.caption("Profit-first pricing recommendations with demand forecasting and trend analytics")

if not Path("artifacts/margin_model.joblib").exists() or not Path("artifacts/demand_model.joblib").exists():
    st.warning(
        "Model artifacts are missing. Run `python -m src.train_margin_model` and "
        "`python -m src.train_demand_model` first."
    )
    st.stop()

bundle = get_bundle()

with st.sidebar:
    st.header("Prediction Inputs")
    category = st.selectbox("Product Category", sorted(data["ProductCategory"].unique().tolist()))
    product_candidates = sorted(data.loc[data["ProductCategory"] == category, "ProductName"].unique().tolist())
    product = st.selectbox("Product", product_candidates)

    product_rows = data[(data["ProductCategory"] == category) & (data["ProductName"] == product)]
    default_brand = product_rows["Brand"].mode().iat[0]
    brand = st.selectbox("Brand", sorted(product_rows["Brand"].unique().tolist()), index=0)

    company = st.selectbox("Marketplace", sorted(data["Company"].unique().tolist()))
    city = st.selectbox("City", sorted(data["City"].unique().tolist()))
    state = st.selectbox("State", sorted(data["State"].unique().tolist()))
    gender = st.selectbox("Customer Gender", sorted(data["CustomerGender"].unique().tolist()))

    customer_age = st.slider("Typical Customer Age", 18, 70, int(data["CustomerAge"].median()))
    unit_price = st.number_input(
        "List Unit Price",
        min_value=1.0,
        value=float(product_rows["UnitPrice"].median()),
        step=10.0,
        format="%.2f",
    )
    shipping = st.number_input("Shipping Fee", min_value=0.0, value=float(product_rows["ShippingFee"].median()), step=5.0)
    rating = st.slider("Expected Rating", 1.0, 5.0, float(product_rows["CustomerRating"].median()), 0.1)
    delivery_days = st.slider("Delivery Days", 0, 30, int(product_rows["DeliveryDays"].median()))

    payment_mode = st.selectbox("Preferred Payment Mode", sorted(data["PaymentMode"].unique().tolist()))
    delivery_status = st.selectbox("Delivery Status Scenario", sorted(data["DeliveryStatus"].unique().tolist()))
    month = st.slider("Order Month", 1, 12, int(data["OrderMonth"].median()))
    day_of_week = st.slider("Order Day Of Week (0=Mon)", 0, 6, int(data["OrderDayOfWeek"].median()))
    is_weekend = int(day_of_week >= 5)
    quarter = (month - 1) // 3 + 1

base_record = {
    "Company": company,
    "CustomerAge": customer_age,
    "CustomerGender": gender,
    "City": city,
    "State": state,
    "ProductCategory": category,
    "ProductName": product,
    "Brand": brand if brand else default_brand,
    "UnitPrice": unit_price,
    "DiscountPercent": float(product_rows["DiscountPercent"].median()),
    "ShippingFee": shipping,
    "PaymentMode": payment_mode,
    "DeliveryStatus": delivery_status,
    "DeliveryDays": float(delivery_days),
    "CustomerRating": float(rating),
    "OrderMonth": int(month),
    "OrderDayOfWeek": int(day_of_week),
    "OrderQuarter": int(quarter),
    "IsWeekend": int(is_weekend),
}

cost_match = cost_lookup[
    (cost_lookup["ProductCategory"] == category) & (cost_lookup["ProductName"] == product) & (cost_lookup["Brand"] == brand)
]
estimated_unit_cost = (
    float(cost_match["MedianUnitCost"].iat[0]) if not cost_match.empty else float(cost_lookup["MedianUnitCost"].median())
)

rec = recommend_margin_and_price(bundle=bundle, base_record=base_record, estimated_unit_cost=estimated_unit_cost, search_points=25)

tradeoff_rows = []
for disc in np.linspace(max(0.0, bundle.discount_min), min(60.0, bundle.discount_max + 2.0), 25):
    probe = dict(base_record)
    probe["DiscountPercent"] = float(disc)
    scenario = evaluate_scenario(bundle=bundle, record=probe, estimated_unit_cost=estimated_unit_cost)
    tradeoff_rows.append(scenario)
tradeoff_df = pd.DataFrame(tradeoff_rows).drop_duplicates(subset=["discount_percent"]).sort_values("discount_percent")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Recommended Discount", f"{rec['discount_percent']:.1f}%")
col2.metric("Predicted Margin", f"{rec['predicted_margin_pct']:.2f}%")
col3.metric("Predicted Demand", f"{rec['predicted_demand_units']:.2f} units")
col4.metric("Expected Profit", f"Rs {rec['expected_total_profit']:.2f}")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Pricing Frontier")
frontier = px.line(
    tradeoff_df,
    x="discount_percent",
    y="expected_total_profit",
    markers=True,
    labels={"discount_percent": "Discount %", "expected_total_profit": "Expected Profit"},
    template="plotly_white",
)
frontier.add_scatter(
    x=[rec["discount_percent"]],
    y=[rec["expected_total_profit"]],
    mode="markers",
    marker=dict(size=12, color="#111111"),
    name="Recommended",
)
st.plotly_chart(frontier, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

latest_month = momentum["Month"].max()
latest = momentum[momentum["Month"] == latest_month].copy()
hot = latest.dropna(subset=["growth_rate"]).sort_values("growth_rate", ascending=False).head(10)
declining = latest.dropna(subset=["growth_rate"]).sort_values("growth_rate", ascending=True).head(10)

v1, v2 = st.columns(2)
with v1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Hot Products Right Now")
    hot_chart = px.bar(
        hot,
        x="growth_rate",
        y="ProductName",
        color="ProductCategory",
        orientation="h",
        template="plotly_white",
        labels={"growth_rate": "Monthly Growth Rate", "ProductName": "Product"},
    )
    hot_chart.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(hot_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with v2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Declining Products")
    dec_chart = px.bar(
        declining,
        x="growth_rate",
        y="ProductName",
        color="ProductCategory",
        orientation="h",
        template="plotly_white",
        labels={"growth_rate": "Monthly Growth Rate", "ProductName": "Product"},
    )
    dec_chart.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(dec_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

month_category = (
    data.assign(Month=data["OrderDate"].dt.to_period("M").dt.to_timestamp())
    .groupby(["Month", "ProductCategory"], as_index=False)["OrderQuantity"]
    .sum()
)
heat = month_category.pivot(index="ProductCategory", columns="Month", values="OrderQuantity").fillna(0)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Demand Heatmap by Category and Month")
heat_fig = px.imshow(
    heat,
    aspect="auto",
    color_continuous_scale="Greys",
    labels=dict(x="Month", y="Category", color="Demand"),
)
st.plotly_chart(heat_fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

state_demand = data.groupby("State", as_index=False)["OrderQuantity"].sum().sort_values("OrderQuantity", ascending=False).head(12)
delivery_impact = (
    data.assign(DeliveryBucket=pd.cut(data["DeliveryDays"], bins=[-1, 2, 5, 8, 15, 45], labels=["0-2", "3-5", "6-8", "9-15", "16+"]))
    .groupby("DeliveryBucket", as_index=False)["OrderQuantity"]
    .mean()
)

v3, v4 = st.columns(2)
with v3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Top Demand States")
    st.plotly_chart(
        px.bar(state_demand, x="State", y="OrderQuantity", template="plotly_white", color="OrderQuantity", color_continuous_scale="Greys"),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with v4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Delivery Speed vs Demand")
    line = go.Figure(
        data=[
            go.Scatter(
                x=delivery_impact["DeliveryBucket"].astype(str),
                y=delivery_impact["OrderQuantity"],
                mode="lines+markers",
                line=dict(color="#111111", width=2),
            )
        ]
    )
    line.update_layout(template="plotly_white", xaxis_title="Delivery Days Bucket", yaxis_title="Average Demand")
    st.plotly_chart(line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Recommendation engine optimizes expected profit by combining predicted margin and predicted demand over a discount search range."
)
