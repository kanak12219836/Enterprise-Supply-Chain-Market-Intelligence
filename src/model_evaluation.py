import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("MODEL EVALUATION")
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

# ==========================================================
# TARGET & FEATURES
# ==========================================================

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
# NUMERIC & CATEGORICAL FEATURES
# ==========================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric Features :", len(numeric_features))
print("Categorical Features :", len(categorical_features))

# ==========================================================
# PREPROCESSING PIPELINE
# ==========================================================

numeric_transformer = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        )

    ]

)

categorical_transformer = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]

)

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )

    ]

)

print("\n✔ Preprocessing Pipeline Ready")

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = joblib.load(MODEL_PATH)

print("\n✔ Trained Model Loaded Successfully")

# ==========================================================
# MAKE PREDICTIONS
# ==========================================================

predictions = model.predict(X_test)

print("\n✔ Predictions Generated")

# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)

plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")

plt.title("Actual vs Predicted")

plt.savefig(
    RESULTS_DIR / "actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✔ Actual vs Predicted Plot Saved")

# ==========================================================
# RESIDUAL PLOT
# ==========================================================

residuals = y_test - predictions

plt.figure(figsize=(8,6))

plt.scatter(
    predictions,
    residuals,
    alpha=0.4
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.savefig(
    RESULTS_DIR / "residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✔ Residual Plot Saved")

# ==========================================================
# ERROR DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,6))

plt.hist(
    residuals,
    bins=40
)

plt.title("Prediction Error Distribution")

plt.xlabel("Residual")

plt.ylabel("Frequency")

plt.savefig(
    RESULTS_DIR / "error_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✔ Error Distribution Saved")

# ==========================================================
# SAVE METRICS
# ==========================================================

metrics_df = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R2 Score"
    ],

    "Value": [
        mae,
        rmse,
        r2
    ]

})

metrics_df.to_csv(

    RESULTS_DIR / "evaluation_metrics.csv",

    index=False

)

print("✔ Evaluation Metrics Saved")

print("\n")
print("=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)