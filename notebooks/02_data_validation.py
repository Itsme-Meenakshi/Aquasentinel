
from pathlib import Path
import pandas as pd

# Locate the project and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
raw_folder = PROJECT_ROOT / "data" / "raw"

files = list(raw_folder.glob("*.xlsx"))

if not files:
    raise FileNotFoundError(
        "No Excel dataset found inside data/raw/"
    )

data_path = files[0]
df = pd.read_excel(data_path)

print("=" * 50)
print("PUFFERFISH DATA VALIDATION")
print("=" * 50)

print("\nDataset:", data_path.name)
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Clean column names
df.columns = df.columns.str.strip()

target = "PUFFER Presence (1/0)"

# 1. Check target labels
print("\n1. PUFFER PRESENCE LABELS")
print(df[target].value_counts(dropna=False))

print("\nUnexpected target values:")
print(df.loc[~df[target].isin([0, 1]), target].value_counts())

# 2. Check dates
print("\n2. DATE VALIDATION")
df["Date_parsed"] = pd.to_datetime(
    df["Date"], format="%d.%m.%Y", errors="coerce"
)

print("Invalid dates:", df["Date_parsed"].isna().sum())
print("Date range:", df["Date_parsed"].min(), "to",
      df["Date_parsed"].max())

# 3. Check missing values
print("\n3. MISSING VALUES")
print(df.isnull().sum())

# 4. Check duplicate rows
print("\n4. DUPLICATES")
print("Duplicate rows:", df.duplicated().sum())

# 5. Check geographical coordinates
print("\n5. COORDINATE VALIDATION")
print("Invalid latitude:",
      ((df["Latitude"] < -90) | (df["Latitude"] > 90)).sum())
print("Invalid longitude:",
      ((df["Longitude"] < -180) | (df["Longitude"] > 180)).sum())

# 6. Check depth and duration
print("\n6. DEPTH AND DURATION")
print("Depth <= 0:", (df["Depth (m)"] <= 0).sum())
print("Duration <= 0:", (df["Duration (Hrs)"] <= 0).sum())

# 7. Check puffer species against presence labels
print("\n7. SPECIES-LABEL CONSISTENCY")
print(pd.crosstab(
    df[target],
    df["Puffer Species"].notna(),
    rownames=["Puffer presence"],
    colnames=["Species recorded"]
))

# 8. Check gear categories
print("\n8. GEAR CATEGORIES")
print(df["Gear"].value_counts(dropna=False))

print("\nValidation complete.")
