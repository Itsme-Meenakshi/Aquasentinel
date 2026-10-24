
from pathlib import Path
import pandas as pd

# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pufferfish_cleaned.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

target = "PUFFER Presence (1/0)"

if target not in df.columns:
    raise KeyError(f"Target column missing: {target}")

print("=" * 60)
print("AQUASENTINEL - FEATURE ASSESSMENT")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("Target distribution:")
print(df[target].value_counts(dropna=False))

# 2. Classify existing features
# These classifications are preliminary and must be verified.
feature_roles = {
    "Date": "Candidate: derive time features",
    "Month": "Candidate: redundant with Date",
    "Season": "Candidate: verify against Date",
    "Latitude": "Candidate: geographic predictor",
    "Longitude": "Candidate: geographic predictor",
    "Depth (m)": "Candidate: verify availability before deployment",
    "Gear": "Candidate: verify availability before deployment",
    "Duration (Hrs)": "Potential leakage: known after fishing",
    "Total catch (Kg)": "Leakage risk: post-haul measurement",
    "Major Catch (Species)": "Leakage risk: catch-derived information",
    target: "TARGET: prediction label",
    "Puffer Species": "Leakage risk: directly linked to target",
    "CPUE(catch per unit effoert) Total Catch (kg/hr)": (
        "Leakage risk: derived from catch and duration"
    ),
}

inventory = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [
        str(df[col].dtype) for col in df.columns
    ],
    "Missing_Count": [
        int(df[col].isna().sum()) for col in df.columns
    ],
    "Unique_Values": [
        int(df[col].nunique(dropna=True)) for col in df.columns
    ],
    "Preliminary_Assessment": [
        feature_roles.get(col, "Review manually")
        for col in df.columns
    ],
})

print("\n--- FEATURE INVENTORY ---")
print(inventory.to_string(index=False))
inventory.to_csv(
    REPORT_DIR / "feature_inventory.csv",
    index=False
)

# 3. Numeric feature summaries by target
numeric_candidates = [
    "Latitude", "Longitude", "Depth (m)",
    "Duration (Hrs)", "Total catch (Kg)",
    "CPUE(catch per unit effoert) Total Catch (kg/hr)"
]

available_numeric = [
    col for col in numeric_candidates
    if col in df.columns
]

print("\n--- NUMERIC FEATURES BY TARGET ---")
numeric_summary = df.groupby(target)[available_numeric].agg(
    ["count", "mean", "median", "std"]
).round(3)

print(numeric_summary)
numeric_summary.to_csv(
    REPORT_DIR / "numeric_features_by_target.csv"
)

# 4. Presence rates for categorical variables
categorical_candidates = ["Season", "Month", "Gear"]

for col in categorical_candidates:
    if col not in df.columns:
        continue

    summary = df.groupby(col, dropna=False).agg(
        Total_Hauls=(target, "size"),
        Positive_Hauls=(target, "sum"),
        Presence_Rate=(target, "mean"),
    )

    summary["Presence_Rate_Percent"] = (
        summary["Presence_Rate"] * 100
    )

    summary = summary.drop(columns="Presence_Rate")

    print(f"\n--- PRESENCE BY {col.upper()} ---")
    print(summary.round(2))

    safe_name = col.lower().replace(" ", "_")
    summary.to_csv(
        REPORT_DIR / f"presence_by_{safe_name}.csv"
    )

# 5. Date consistency checks
if "Date" in df.columns:
    dates = pd.to_datetime(df["Date"], errors="coerce")

    print("\n--- DATE CONSISTENCY ---")
    print("Invalid dates:", int(dates.isna().sum()))

    if "Month" in df.columns:
        expected_month = dates.dt.month_name()
        actual_month = df["Month"].astype("string").str.strip()

        mismatch = (
            expected_month.str.casefold()
            != actual_month.str.casefold()
        )

        print(
            "Month/date mismatches:",
            int(mismatch.fillna(False).sum())
        )

    if "Season" in df.columns:
        print("\nSeason values:")
        print(df["Season"].value_counts(dropna=False))

        print(
            "Note: verify season definitions with the data "
            "provider before changing labels."
        )

# 6. Record the environmental attributes to investigate
new_features = pd.DataFrame([
    {
        "Proposed_Feature": "Sea surface temperature",
        "Potential_Source": "Copernicus Marine / NASA",
        "Purpose": "Describe thermal conditions",
        "Status": "To source and validate",
    },
    {
        "Proposed_Feature": "Sea surface salinity",
        "Potential_Source": "Copernicus Marine",
        "Purpose": "Describe surface-water conditions",
        "Status": "To source and validate",
    },
    {
        "Proposed_Feature": "Chlorophyll-a",
        "Potential_Source": "Copernicus Marine / NASA Ocean Color",
        "Purpose": "Proxy for ocean productivity",
        "Status": "To source and validate",
    },
    {
        "Proposed_Feature": "Ocean current components",
        "Potential_Source": "Copernicus Marine",
        "Purpose": "Describe water movement",
        "Status": "To source and validate",
    },
    {
        "Proposed_Feature": "Dissolved oxygen",
        "Potential_Source": "Copernicus Marine, if suitable coverage exists",
        "Purpose": "Describe marine habitat conditions",
        "Status": "To source and validate",
    },
    {
        "Proposed_Feature": "Wave height",
        "Potential_Source": "Copernicus Marine",
        "Purpose": "Describe sea state",
        "Status": "To source and validate",
    },
])

new_features.to_csv(
    REPORT_DIR / "proposed_environmental_features.csv",
    index=False
)

print("\n--- PROPOSED NEW FEATURES ---")
print(new_features.to_string(index=False))

print("\nReports saved in:", REPORT_DIR)
print("Feature assessment complete.")
