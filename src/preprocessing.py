import pandas as pd
from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "cleaned_supply_chain.csv"


# ==========================================================
# Preprocessing Function
# ==========================================================
def preprocess_data():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    # Load Dataset
    df = pd.read_csv(
        RAW_DATA,
        encoding="latin1",
        low_memory=False
    )

    print(f"\nOriginal Shape : {df.shape}")

    # ------------------------------------------------------
    # Missing Values Before Cleaning
    # ------------------------------------------------------
    print("\nMissing Values Before Cleaning:\n")
    print(df.isnull().sum())

    # ------------------------------------------------------
    # Duplicate Rows
    # ------------------------------------------------------
    duplicates = df.duplicated().sum()

    print(f"\nDuplicate Rows : {duplicates}")

    df = df.drop_duplicates()

    # ------------------------------------------------------
    # Remove Completely Empty Columns
    # ------------------------------------------------------
    df = df.dropna(axis=1, how="all")

    # ------------------------------------------------------
    # Fill Numeric Columns
    # ------------------------------------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    # ------------------------------------------------------
    # Fill Categorical Columns
    # ------------------------------------------------------
    categorical_cols = df.select_dtypes(exclude=["number"]).columns

    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # ------------------------------------------------------
    # Convert Zipcodes to String
    # ------------------------------------------------------
    if "Customer Zipcode" in df.columns:
        df["Customer Zipcode"] = df["Customer Zipcode"].astype(str)

    if "Order Zipcode" in df.columns:
        df["Order Zipcode"] = df["Order Zipcode"].astype(str)

    # ------------------------------------------------------
    # Missing Values After Cleaning
    # ------------------------------------------------------
    print("\nMissing Values After Cleaning:\n")
    print(df.isnull().sum())

    print(f"\nFinal Shape : {df.shape}")

    # ------------------------------------------------------
    # Save Clean Dataset
    # ------------------------------------------------------
    
    print("\nSaving cleaned dataset...")
    df.to_csv(PROCESSED_DATA, index=False)

    print("\n" + "=" * 60)
    print("Cleaned Dataset Saved Successfully!")
    print(PROCESSED_DATA)
    print("=" * 60)


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    preprocess_data()