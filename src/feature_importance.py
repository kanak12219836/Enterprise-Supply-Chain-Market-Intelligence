import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_engineered_supply_chain.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_model.pkl"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

RESULTS_DIR.mkdir(exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================


df = pd.read_csv(DATA_PATH)


print("\nDataset Shape :", df.shape)

target = "Order Item Quantity"

y = df[target]

X = df.drop(columns=[target])

# ==========================================================
# REMOVE LEAKAGE FEATURES
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

X = X.drop(

    columns=[c for c in leakage_features if c in X.columns]

)

print("\nAfter Leakage Removal :", X.shape)

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

X = X.drop(

    columns=[c for c in high_cardinality_columns if c in X.columns]

)

print("After Removing High Cardinality :", X.shape)

# ==========================================================
# REMOVE PERSONAL COLUMNS
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

    "Product Card Id"

]

X = X.drop(

    columns=[c for c in drop_cols if c in X.columns]

)

print("Final Feature Shape :", X.shape)


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

pipeline = joblib.load(MODEL_PATH)

print("✔ Trained Model Loaded")


# ==========================================================
# EXTRACT PREPROCESSOR & MODEL
# ==========================================================

preprocessor = pipeline.named_steps["preprocessor"]

model = pipeline.named_steps["model"]

print("✔ Preprocessor Extracted")
print("✔ Random Forest Extracted")

# ==========================================================
# GET FEATURE NAMES
# ==========================================================

feature_names = preprocessor.get_feature_names_out()
feature_importance = model.feature_importances_

print("\nFeature Names Length :", len(feature_names))
print("Feature Importance Length :", len(feature_importance))

# ==========================================================
# CREATE FEATURE IMPORTANCE DATAFRAME
# ==========================================================

importance_df = pd.DataFrame({

    "Feature": feature_names,
    "Importance": feature_importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop 20 Important Features:\n")

print(importance_df.head(20))

# ==========================================================
# SAVE CSV
# ==========================================================

importance_df.to_csv(

    RESULTS_DIR / "feature_importance.csv",

    index=False

)
print("✔ Feature Importance CSV Saved")

# ==========================================================
# FEATURE IMPORTANCE PLOT
# ==========================================================

top20 = importance_df.head(20)

plt.figure(figsize=(10,8))

plt.barh(

    top20["Feature"],

    top20["Importance"]

)

plt.gca().invert_yaxis()

plt.xlabel("Importance")

plt.ylabel("Features")

plt.title("Top 20 Feature Importance")

plt.tight_layout()

plt.savefig(

    RESULTS_DIR / "feature_importance.png",

    dpi=300

)

plt.close()

print("✔ Feature Importance Plot Saved")

print("\nResults Saved:")
print("•", RESULTS_DIR / "feature_importance.csv")
print("•", RESULTS_DIR / "feature_importance.png")

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE COMPLETED")
print("=" * 60)

