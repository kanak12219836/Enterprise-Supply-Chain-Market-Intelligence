import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("HYPERPARAMETER TUNING")
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

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(DATA_PATH)

print("\nOriginal Dataset Shape :", df.shape)

# Sample dataset for faster tuning 
# Use a subset for faster hyperparameter tuning.
# Final model should be trained on the complete dataset.
df = df.sample(
    n=50000,
    random_state=42
).reset_index(drop=True)

print("Sample Dataset Shape   :", df.shape)

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
    "Product Card Id",
]

X = X.drop(
    columns=[c for c in drop_cols if c in X.columns]
)

print("Final Feature Shape :", X.shape)

print("\nColumns Used:")
print("-" * 40)
for i, col in enumerate(X.columns, start=1):
    print(f"{i}. {col}")

# ==========================================================
# DATE HANDLING
# ==========================================================

date_cols = [
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]

for col in date_cols:
    if col in X.columns:
        X[col] = pd.to_datetime(
            X[col],
            errors="coerce"
        )

        X[col] = (
            X[col].astype("int64") // 10**9
        )


# ==========================================================
# FEATURE BREAKDOWN
# ==========================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

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

print("\n✔ Part 1 Complete (Leakage + Cleaning Done)")

# ==========================================================
# PREPROCESSING PIPELINE
# ==========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
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

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ==========================================================
# RANDOM FOREST PIPELINE
# ==========================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\n✔ Pipeline Created")

# ==========================================================
# HYPERPARAMETER SEARCH SPACE
# ==========================================================

param_grid = {

    "model__n_estimators": [
        100,
        200,
        300
    ],

    "model__max_depth": [
        10,
        20,
        30,
        None
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ]

}

print("\n✔ Parameter Grid Ready")

# ==========================================================
# RANDOMIZED SEARCH
# ==========================================================

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_grid,

    n_iter=10,

    cv=3,

    scoring="r2",

    random_state=42,

    n_jobs=-1,

    verbose=2

)

print("\n✔ RandomizedSearchCV Ready")

# ==========================================================
# START HYPERPARAMETER TUNING
# ==========================================================

print("\n")
print("=" * 60)
print("STARTING HYPERPARAMETER TUNING")
print("=" * 60)

search.fit(
    X_train,
    y_train
)

print("\n✔ Hyperparameter Tuning Completed")

# ==========================================================
# BEST PARAMETERS
# ==========================================================

print("\nBest Parameters Found")
print("-" * 40)

for key, value in search.best_params_.items():
    print(f"{key}: {value}")

print(f"\nBest CV Score : {search.best_score_:.4f}")

# ==========================================================
# BEST MODEL
# ==========================================================

best_model = search.best_estimator_

predictions = best_model.predict(X_test)

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
print("TUNED MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")

# ==========================================================
# SAVE TUNED MODEL
# ==========================================================

joblib.dump(
    best_model,
    MODEL_DIR / "tuned_random_forest.pkl"
)

print("\n" + "=" * 60)
print("TUNED MODEL SAVED")
print("=" * 60)

print("\nModel Saved At:")
print(MODEL_DIR / "tuned_random_forest.pkl")

