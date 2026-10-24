
from pathlib import Path
import pandas as pd

# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "processed"

PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

# 2. Load the original Excel dataset
excel_files = list(RAW_FOLDER.glob("*.xlsx"))

if not excel_files:
    raise FileNotFoundError(
        "No Excel dataset found inside data/raw/"
    )

data_path = excel_files[0]
df = pd.read_excel(data_path)

print("=" * 55)
print("AQUASENTINEL - DATA CLEANING")
print("=" * 55)
print("Original dataset:", data_path.name)
print("Original shape:", df.shape)

# 3. Standardize column names
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Parse and preserve the original date and month
df["Month_original"] = df["Month"]

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d.%m.%Y",
    errors="coerce"
)

# Derive month from the parsed date
df["Month"] = df["Date"].dt.month_name()

# 5. Standardize text fields
text_columns = [
    "Month",
    "Season",
    "Gear",
    "Major Catch (Species)",
    "Puffer Species"
]

for column in text_columns:
    if column in df.columns:
        df[column] = df[column].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

# 6. Standardize verified gear-name variations
# These two labels differ only by capitalization.
df["Gear"] = df["Gear"].replace({
    "Mini Ring Seine": "Mini ring seine"
})

# 7. Check target labels
target = "PUFFER Presence (1/0)"

if target not in df.columns:
    raise KeyError(f"Target column not found: {target}")

print("\nTarget distribution:")
print(df[target].value_counts(dropna=False))

unexpected_labels = ~df[target].isin([0, 1])

print("\nUnexpected or missing target labels:",
      int(unexpected_labels.sum()))

# 8. Validate dates and essential fields
print("\nInvalid dates:", int(df["Date"].isna().sum()))

essential_columns = [
    "Date", "Latitude", "Longitude",
    "Depth (m)", "Gear", target
]

print("\nMissing values in essential columns:")
print(df[essential_columns].isna().sum())

# 9. Check duplicates; report without deleting automatically
print("\nExact duplicate rows:", int(df.duplicated().sum()))

# 10. Check coordinate and measurement ranges
print("\nInvalid latitude:",
      int((~df["Latitude"].between(-90, 90)).sum()))

print("Invalid longitude:",
      int((~df["Longitude"].between(-180, 180)).sum()))

print("Non-positive depth:",
      int((df["Depth (m)"] <= 0).sum()))

print("Non-positive duration:",
      int((df["Duration (Hrs)"] <= 0).sum()))

# 11. Save the cleaned copy
# No rows are deleted and no missing values are imputed here.
output_path = PROCESSED_FOLDER / "pufferfish_cleaned.csv"
df.to_csv(output_path, index=False)

# 12. Save a cleaning report
report_path = PROCESSED_FOLDER / "cleaning_report.txt"

with open(report_path, "w", encoding="utf-8") as report:
    report.write("AQUASENTINEL DATA CLEANING REPORT\n")
    report.write("=" * 40 + "\n")
    report.write(f"Source file: {data_path.name}\n")
    report.write(f"Original rows: {len(df)}\n")
    report.write(f"Columns: {len(df.columns)}\n")
    report.write(f"Duplicate rows: {df.duplicated().sum()}\n")
    report.write(f"Invalid dates: {df['Date'].isna().sum()}\n")
    report.write("\nMissing values:\n")
    report.write(df.isna().sum().to_string())
    report.write("\n\nTarget distribution:\n")
    report.write(df[target].value_counts(dropna=False).to_string())

print("\nCleaned dataset saved to:")
print(output_path)

print("\nCleaning report saved to:")
print(report_path)

print("\nFinal shape:", df.shape)
print("Data cleaning completed.")
