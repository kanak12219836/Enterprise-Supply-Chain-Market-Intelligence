import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_supply_chain.csv"
DOCS_PATH = PROJECT_ROOT / "docs"

# Create docs folder if it doesn't exist
DOCS_PATH.mkdir(exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================
df = pd.read_csv(DATA_PATH, low_memory=False)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ==========================================================
# Top 10 Countries by Orders
# ==========================================================
top_countries = df["Order Country"].value_counts().head(10)

print("\nTop 10 Countries by Orders:")
print(top_countries)

plt.figure(figsize=(12, 6))
top_countries.plot(kind="bar")

plt.title("Top 10 Countries by Orders")
plt.xlabel("Country")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(DOCS_PATH / "top_countries.png")
plt.show()

# ==========================================================
# Shipping Mode Distribution
# ==========================================================
shipping = df["Shipping Mode"].value_counts()

print("\nShipping Mode Distribution:")
print(shipping)

plt.figure(figsize=(8, 6))
shipping.plot(kind="pie", autopct="%1.1f%%")

plt.ylabel("")
plt.title("Shipping Mode Distribution")

plt.tight_layout()
plt.savefig(DOCS_PATH / "shipping_mode.png")
plt.show()

# ==========================================================
# Top Product Categories
# ==========================================================
top_categories = df["Category Name"].value_counts().head(10)

print("\nTop Product Categories:")
print(top_categories)

plt.figure(figsize=(12, 6))
top_categories.plot(kind="bar")

plt.title("Top 10 Product Categories")
plt.xlabel("Category")
plt.ylabel("Orders")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(DOCS_PATH / "top_categories.png")
plt.show()

print("\nEDA Completed Successfully!")