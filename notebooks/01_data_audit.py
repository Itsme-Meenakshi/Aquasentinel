
from pathlib import Path
import pandas as pd
import numpy as np

# Find the root of our project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Find Excel files in the raw data folder
raw_folder = PROJECT_ROOT / "data" / "raw"
excel_files = list(raw_folder.glob("*.xlsx"))

if not excel_files:
    raise FileNotFoundError(
        f"No Excel file found in {raw_folder}. "
        "Copy the CMFRI dataset into data/raw first."
    )

data_path = excel_files[0]
df = pd.read_excel(data_path)

print("DATA AUDIT — PUFFERFISH PROJECT")
print("=" * 45)

print("\nDataset file:", data_path.name)
print("Rows and columns:", df.shape)

print("\nColumn names:")
for column in df.columns:
    print("-", repr(column))

print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nData types:")
print(df.dtypes.to_string())

print("\nMissing values per column:")
print(df.isna().sum().to_string())

print("\nCompletely duplicated rows:", df.duplicated().sum())
