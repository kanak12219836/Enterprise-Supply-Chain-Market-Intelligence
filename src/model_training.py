import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from pathlib import Path

# ==========================================================
# PATH SETUP
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_engineered_supply_chain.csv"
)

print("=" * 60)
print("SUPPLY CHAIN MODEL TRAINING PIPELINE")
print("=" * 60)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"\nDataset Shape : {df.shape}")

TARGET = "Order Item Quantity"

# ==========================================================
# SPLIT FEATURES & TARGET (TEMP)
# ==========================================================

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ==========================================================
# LEAKAGE CHECK (IMPORTANT DEBUG STEP)
# ==========================================================

print("\nChecking correlation with target...\n")

temp = X.copy()
temp[TARGET] = y

corr = temp.corr(numeric_only=True)[TARGET].sort_values(ascending=False)

print("Top correlated features with target:")
print(corr.head(15))
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

# ==========================================================
# FEATURE TYPE SPLIT
# ==========================================================

# Select Features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()


print("\nFeature Breakdown")
print("-" * 40)
print("Numeric Features     :", len(numeric_features))
print("Categorical Features :", len(categorical_features))

print("\nNumeric Columns:")
for col in numeric_features:
    print(col)

print("\nCategorical Columns:")
for col in categorical_features:
    print(col)

print("\n Part 1 Complete (Leakage + Cleaning Done)")

print("\nTarget Summary")
print("-" * 40)

print(y.describe())

print("\nUnique Values:", y.nunique())

print("\nTop 20 Target Values")

print(y.value_counts().head(20))

print("\nColumns Currently Used")
print("-" * 40)

for i, col in enumerate(X.columns, start=1):
    print(f"{i}. {col}")

# ==========================================================
# IMPORT ML LIBRARIES
# ==========================================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
# ==========================================================
# PREPROCESSING PIPELINE
# ==========================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

print("\n✔ Preprocessing Pipeline Ready")
# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")
# ==========================================================
# BUILD MODEL PIPELINE
# ==========================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

print("\n Machine Learning Pipeline Created")
# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\nTraining Random Forest Model...\n")

pipeline.fit(X_train, y_train)

print(" Model Training Completed")
# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = pipeline.predict(X_test)

# ==========================================================
# EVALUATION
# ==========================================================

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

print("=" * 60)
# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": pipeline.named_steps["model"].feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 60)

print(importance.head(20))
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

importance.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)

import matplotlib.pyplot as plt

top20 = importance.head(20)

plt.figure(figsize=(10, 8))

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

metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2]
})

metrics.to_csv(
    RESULTS_DIR / "model_metrics.csv",
    index=False
)

# ==========================================================
# SAVE TRAINED MODEL
# ==========================================================

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    pipeline,
    MODEL_DIR / "random_forest_model.pkl"
)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nModel Saved At:")
print(MODEL_DIR / "random_forest_model.pkl")

print("\nResults Saved:")

print("•", RESULTS_DIR / "feature_importance.csv")
print("•", RESULTS_DIR / "feature_importance.png")
print("•", RESULTS_DIR / "model_metrics.csv")

print("=" * 60)