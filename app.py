import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import streamlit as st

from pathlib import Path

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="Supply Chain Demand Prediction",

    page_icon="📦",

    layout="wide"

)

st.title("📦 Supply Chain Demand Prediction")

st.write(
    "Predict Order Item Quantity using Machine Learning."
)

# ==========================================================
# LOAD MODEL
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_model.pkl"
)

model = joblib.load(MODEL_PATH)

st.sidebar.header("Input Features")

# ==========================================================
# USER INPUTS
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    order_type = st.selectbox(
        "Type",
        ["DEBIT", "TRANSFER", "PAYMENT", "CASH"]
    )

    shipping_real = st.number_input(
        "Days for Shipping (Real)",
        min_value=0,
        value=4
    )

    shipping_scheduled = st.number_input(
        "Days for Shipment (Scheduled)",
        min_value=0,
        value=4
    )

    delivery_status = st.selectbox(
        "Delivery Status",
        [
            "Advance shipping",
            "Late delivery",
            "Shipping canceled",
            "Shipping on time"
        ]
    )

    late_delivery = st.selectbox(
        "Late Delivery Risk",
        [0, 1]
    )

with col2:

    category_id = st.number_input(
        "Category ID",
        value=17
    )

    product_price = st.number_input(
        "Product Price",
        value=327.75
    )

    shipping_delay = st.number_input(
        "Shipping Delay",
        value=0
    )

    order_year = st.number_input(
        "Order Year",
        value=2017
    )

    order_month = st.number_input(
        "Order Month",
        value=1
    )

    customer_country = st.text_input(
        "Customer Country",
        value="USA"
    )

    customer_segment = st.selectbox(
        "Customer Segment",
        ["Consumer", "Corporate", "Home Office"]
    )

    department_id = st.number_input(
        "Department ID",
        value=4
    )

    department_name = st.text_input(
        "Department Name",
        value="Apparel"
    )

    category_name = st.text_input(
        "Category Name",
        value="Cleats"
    )

    latitude = st.number_input(
        "Latitude",
        value=18.25
    )

    longitude = st.number_input(
        "Longitude",
        value=-66.03
    )

    market = st.text_input(
        "Market",
        value="Pacific Asia"
    )

    order_country = st.text_input(
        "Order Country",
        value="Indonesia"
    )

    order_region = st.text_input(
        "Order Region",
        value="Southeast Asia"
    )

    order_status = st.selectbox(
        "Order Status",
        ["COMPLETE", "PENDING", "PROCESSING", "CLOSED"]
    )

    product_category_id = st.number_input(
        "Product Category ID",
        value=17
    )

    product_status = st.selectbox(
        "Product Status",
        [0, 1]
    )

    order_item_cardprod_id = st.number_input(
        "Order Item Cardprod ID",
        value=1360
    )

    order_item_profit_ratio = st.number_input(
        "Order Item Profit Ratio",
        value=0.29,
        format="%.2f"
    )

    order_date = st.number_input(
        "Order Date (YYYYMMDD)",
        value=20170115
    )

    shipping_date = st.number_input(
        "Shipping Date (YYYYMMDD)",
        value=20170119
    )

    order_day = st.number_input(
        "Order Day",
        value=15
    )

    order_weekday = st.selectbox(
        "Order Weekday",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    shipping_mode = st.selectbox(
        "Shipping Mode",
        [
            "Standard Class",
            "Second Class",
            "First Class",
            "Same Day"
        ]
    )

# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

input_data = {

    "Type": order_type,

    "Days for shipping (real)": shipping_real,

    "Days for shipment (scheduled)": shipping_scheduled,

    "Delivery Status": delivery_status,

    "Late_delivery_risk": late_delivery,

    "Category Id": category_id,

    "Category Name": category_name,

    "Customer Country": customer_country,

    "Customer Segment": customer_segment,

    "Department Id": department_id,

    "Department Name": department_name,

    "Latitude": latitude,

    "Longitude": longitude,

    "Market": market,

    "Order Country": order_country,

    "order date (DateOrders)": order_date,

    "Order Item Cardprod Id": order_item_cardprod_id,

    "Order Item Profit Ratio": order_item_profit_ratio,

    "Order Region": order_region,

    "Order Status": order_status,

    "Product Category Id": product_category_id,

    "Product Price": product_price,

    "Product Status": product_status,

    "shipping date (DateOrders)": shipping_date,

    "Shipping Mode": shipping_mode,

    "Order Year": order_year,

    "Order Month": order_month,

    "Order Day": order_day,

    "Order Weekday": order_weekday,

    "Shipping Delay": shipping_delay

}

input_df = pd.DataFrame([input_data])

# ==========================================================
# PREDICT BUTTON
# ==========================================================

if st.button("Predict Order Quantity"):

    prediction = model.predict(input_df)

    st.success(
        f"Predicted Order Quantity : {prediction[0]:.2f}"
    )