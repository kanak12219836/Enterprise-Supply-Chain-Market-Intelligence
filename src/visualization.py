# imports
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import r2_score

#Paths
print("=" * 60)
print("SUPPLY CHAIN VISUALIZATION")
print("=" * 60)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "feature_engineered_supply_chain.csv"

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

#Load Dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print(f"Dataset Shape : {df.shape}")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

#Target
target = "Order Item Quantity"

y = df[target]

X = df.drop(columns=[target])

# ==========================================================
# REMOVE LEAKAGE FEATURES (VERY IMPORTANT)
# ==========================================================

leakage_features = [
    "Sales",
    "Sales per customer",
    "Order Item Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Total",
    "Order Profit Per Order",
    "Benefit per order",
    "Profit Margin"
]

X = X.drop(columns=[c for c in leakage_features if c in X.columns])

print("\nAfter Leakage Removal:")
print(f"Features Shape : {X.shape}")

# ==========================================================
# REMOVE HIGH CARDINALITY COLUMNS
# ==========================================================

high_cardinality_columns = [
    "Product Name",
    "Product Image",
    "Customer City",
    "Order City",
    "Customer State",
    "Order State",
    "Customer Zipcode",
    "Order Zipcode"
]

X = X.drop(columns=[c for c in high_cardinality_columns if c in X.columns])

print("\nAfter High Cardinality Removal:")
print(f"Features Shape : {X.shape}")

# ==========================================================
# REMOVE ID / PII COLUMNS
# ==========================================================

drop_cols = [
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Product Card Id",

]

X = X.drop(columns=[c for c in drop_cols if c in X.columns])

print("\nAfter Removing Unnecessary Columns:")
print(f"Features Shape : {X.shape}")


# ==========================================================
# DATE HANDLING
# ==========================================================

date_cols = [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

for col in date_cols:
    if col in X.columns:
        X[col] = pd.to_datetime(X[col], errors="coerce")
        X[col] = X[col].astype("int64") // 10**9
        
#Load Model
model = joblib.load(MODEL_PATH)

print("\n✔ Trained Model Loaded")

#Prediction
pred = model.predict(X)

print("✔ Predictions Generated")

#Graph 1

plt.figure(figsize=(8,5))

sns.countplot(
    x=y,
    hue=y,
    palette="viridis",
    legend=False
)

plt.title("Order Quantity Distribution")
plt.xlabel("Order Quantity")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR/"order_quantity_distribution.png",
    dpi=300
)

plt.close()
print(" Chart 1 Saved")

#Graph 2
plt.figure(figsize=(8,5))

sns.histplot(
    df["Product Price"],
    bins=40,
    kde=True,
    color="orange"
)

plt.title("Product Price Distribution")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR/"product_price_distribution.png",
    dpi=300
)

plt.close()
print(" Chart 2 Saved")

#Graph 3
numeric = df.select_dtypes(include="number")

corr = numeric.corr().round(2)

plt.figure(figsize=(12,10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.2,
    square=True,
     cbar_kws={"shrink":0.8}
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR/"correlation_heatmap.png",
    dpi=300
)

plt.close()
print(" Chart 3 Saved")

#Graph 4
top = (
    df["Category Name"]
      .value_counts()
      .head(10)
)

plt.figure(figsize=(10,6))

top.plot(
    kind="bar",
    color="steelblue",
    edgecolor="black"
)

plt.title("Top 10 Product Categories")
plt.ylabel("Orders")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR/"top_categories.png",
    dpi=300
)

plt.close()
print(" Chart 4 Saved")

#Graph 5
plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Shipping Mode",
    hue="Shipping Mode",
    palette="Set2",
    legend=False
)

plt.title("Shipping Mode Distribution")
plt.xlabel("Shipping Mode")
plt.ylabel("Orders")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR/"shipping_mode_distribution.png",
    dpi=300
)

plt.close()
print(" Chart 5 Saved")

#Graph 6
plt.figure(figsize=(7,7))

plt.scatter(
    y,
    pred,
    alpha=0.25,
    s=10,
    color="royalblue"
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    color="red",
    linestyle="--"
)

plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")
plt.title("Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR/"actual_vs_predicted.png",
    dpi=300
)

plt.close()
print(" Chart 6 Saved")

#Graph 7 (Feature Importance)
importance = pd.read_csv(
    RESULTS_DIR / "feature_importance.csv"
)

top20 = importance.head(20)

plt.figure(figsize=(14,10))

plt.barh(
    top20["Feature"],
    top20["Importance"],
    color="teal"
)

plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 20 Important Features")

plt.subplots_adjust(left=0.42)

plt.grid(axis="x", linestyle="--", alpha=0.3)

plt.savefig(
    RESULTS_DIR/"feature_importance_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
print("Chart 7 Saved")

score = r2_score(y, pred)

print(f"\nR² Score : {score:.4f}")

print("\nResults Saved:")

print("•", RESULTS_DIR/"order_quantity_distribution.png")
print("•", RESULTS_DIR/"product_price_distribution.png")
print("•", RESULTS_DIR/"correlation_heatmap.png")
print("•", RESULTS_DIR/"top_categories.png")
print("•", RESULTS_DIR/"shipping_mode_distribution.png")
print("•", RESULTS_DIR/"actual_vs_predicted.png")
print("•", RESULTS_DIR/"feature_importance_visualization.png")

print("\nTotal Visualizations Saved : 7")

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 60)