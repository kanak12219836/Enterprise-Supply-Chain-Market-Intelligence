import pandas as pd
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"


def load_data():
    """
    Load the supply chain dataset.
    """
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    return df


if __name__ == "__main__":
    df = load_data()

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nFirst 5 Rows:\n")
    print(df.head())