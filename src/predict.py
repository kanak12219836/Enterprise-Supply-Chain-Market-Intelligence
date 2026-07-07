import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from pathlib import Path

print("=" * 60)
print("SUPPLY CHAIN DEMAND PREDICTION")
print("=" * 60)

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_model.pkl"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(MODEL_PATH)

print("\n✔ Trained Model Loaded")

# ==========================================================
# SAMPLE INPUT
# ==========================================================

sample_data = {

    "Type": "DEBIT",

    "Days for shipping (real)": 4,

    "Days for shipment (scheduled)": 4,

    "Delivery Status": "Advance shipping",

    "Late_delivery_risk": 0,

    "Category Id": 17,

    "Category Name": "Cleats",

    "Customer Country": "USA",

    "Customer Segment": "Consumer",

    "Department Id": 4,

    "Department Name": "Apparel",

    "Latitude": 18.25,

    "Longitude": -66.03,

    "Market": "Pacific Asia",

    "Order Country": "Indonesia",

    "order date (DateOrders)": "2017-01-15",
    
    "shipping date (DateOrders)": "2017-01-19",

    "Order Item Cardprod Id": 1360,

    "Order Item Profit Ratio": 0.29,

    "Order Region": "Southeast Asia",

    "Order Status": "COMPLETE",

    "Product Category Id": 17,

    "Product Price": 327.75,

    "Product Status": 0,

    "Shipping Mode": "Standard Class",

    "Order Year": 2017,

    "Order Month": 1,

    "Order Day": 15,

    "Order Weekday": "Sunday",
    
    "Weekend Order": 1,

    "Shipping Delay": 0

}
# ==========================================================
# DISPLAY SETTINGS
# ==========================================================

SHOW_INPUT = True

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_order(input_data):

    input_df = pd.DataFrame([input_data])

    # Convert date columns (same as model_training.py)
    date_cols = [
        "order date (DateOrders)",
        "shipping date (DateOrders)"
    ]

    for col in date_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_datetime(
                input_df[col],
                errors="coerce"
            )

            input_df[col] = (
                input_df[col].astype("int64") // 10**9
            )

    prediction = model.predict(input_df)

    return round(prediction[0])

# ==========================================================
# MAKE PREDICTION
# ==========================================================

predicted_quantity = predict_order(sample_data)

if SHOW_INPUT:
    print("\nInput Data:")
    print(pd.DataFrame([sample_data]))

print("\n")
print("=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(f"Predicted Order Quantity : {predicted_quantity}")

print("\n" + "=" * 60)
print("PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)
