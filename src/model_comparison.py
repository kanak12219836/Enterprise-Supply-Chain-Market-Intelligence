import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_engineered_supply_chain.csv"
)

# Read Dataset
df = pd.read_csv(DATA_PATH)

print("\nDataset Shape :", df.shape)

# ==========================================================
# SAMPLE DATA FOR MODEL COMPARISON
# ==========================================================

df = df.sample(
    n=50000,
    random_state=42
).reset_index(drop=True)

print("\nUsing Sample Dataset :", df.shape)

target = "Order Item Quantity"

y = df[target]

X = df.drop(columns=[target])

leakage_features = [
    "Sales per customer",
    "Order Item Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Total",
    "Order Profit Per Order",
    "Benefit per order",
    "Profit Margin",
    "Sales"
]

X = X.drop(
    columns=[c for c in leakage_features if c in X.columns]
)

print("\nAfter Leakage Removal :", X.shape)

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

print("After Removing Personal Columns :", X.shape)

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

print("Final Feature Shape :", X.shape)

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nNumeric Features :", len(numeric_features))
print("Categorical Features :", len(categorical_features))

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
        ("cat", categorical_pipeline, categorical_features)
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
# MODELS
# ==========================================================

models = {

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )

}


print("\nModels Ready:")
for model_name in models.keys():
    print("✔", model_name)
    
# ==========================================================
# TRAIN MODELS
# ==========================================================

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2 Score": round(r2, 4)
    })

    print(f"✔ {model_name} Completed")
    
# ==========================================================
# MODEL COMPARISON RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print(results_df)

print("\nBest Model :")
print(results_df.iloc[0]["Model"])

# ==========================================================
# SAVE RESULTS
# ==========================================================

results_path = (
    BASE_DIR
    / "results"
)

results_path.mkdir(exist_ok=True)

results_df.to_csv(
    results_path / "model_comparison_results.csv",
    index=False
)

print("\n✔ Results Saved Successfully")