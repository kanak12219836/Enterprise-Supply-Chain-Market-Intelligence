import pandas as pd
from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_supply_chain.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_supply_chain.csv"

# -----------------------------
# Load Dataset
# -----------------------------
print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print(f"\nOriginal Shape : {df.shape}")

# -----------------------------
# Convert Date Columns
# -----------------------------
df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

df["shipping date (DateOrders)"] = pd.to_datetime(
    df["shipping date (DateOrders)"],
    errors="coerce"
)

# -----------------------------
# Extract Date Features
# -----------------------------
df["Order Year"] = df["order date (DateOrders)"].dt.year
df["Order Month"] = df["order date (DateOrders)"].dt.month
df["Order Day"] = df["order date (DateOrders)"].dt.day
df["Order Weekday"] = df["order date (DateOrders)"].dt.day_name()

# -----------------------------
# Weekend Order
# -----------------------------
df["Weekend Order"] = (
    df["Order Weekday"]
        .isin(["Saturday", "Sunday"])
        .astype(int)
)

# -----------------------------
# Shipping Delay
# -----------------------------
df["Shipping Delay"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

# -----------------------------
# Profit Margin (%)
# -----------------------------
import numpy as np

df["Profit Margin"] = np.where(
    df["Sales"] != 0,
    (df["Order Profit Per Order"] / df["Sales"]) * 100,
    0
)

df["Profit Margin"] = df["Profit Margin"].fillna(0)

# -----------------------------
# Dataset Summary
# -----------------------------

print(f"\nNew Shape : {df.shape}")

print("\nMissing Values (Top 10):")
print(
    df.isnull()
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print("\nDuplicate Rows :", df.duplicated().sum())

print("\nDataset Information:")
df.info()

# -----------------------------
# Save Dataset
# -----------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature Engineering Completed Successfully!")

print(f"\nNew Shape : {df.shape}")

print("\nNew Features Added:")

new_features = [
    "Order Year",
    "Order Month",
    "Order Day",
    "Order Weekday",
    "Weekend Order",
    "Shipping Delay",
    "Profit Margin"
]

for feature in new_features:
    print("✔", feature)

print("\nSaved To:")
print(OUTPUT_PATH)

print("=" * 60)