
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
raw_folder = PROJECT_ROOT / "data" / "raw"
report_folder = PROJECT_ROOT / "reports"
report_folder.mkdir(exist_ok=True)

files = list(raw_folder.glob("*.xlsx"))
if not files:
    raise FileNotFoundError("No Excel dataset found in data/raw/")

df = pd.read_excel(files[0])
df.columns = df.columns.str.strip()

target = "PUFFER Presence (1/0)"
df["Date"] = pd.to_datetime(
    df["Date"], format="%d.%m.%Y", errors="coerce"
)

print("\n--- TARGET DISTRIBUTION ---")
print(df[target].value_counts())
print("\nPresence rate:", round(df[target].mean() * 100, 2), "%")

print("\n--- PRESENCE BY SEASON ---")
print(pd.crosstab(
    df["Season"], df[target],
    margins=True,
    normalize="index"
).round(3))

print("\n--- PRESENCE BY GEAR ---")
print(pd.crosstab(
    df["Gear"], df[target],
    margins=True
))

print("\n--- NUMERIC FEATURE SUMMARY ---")
numeric_cols = [
    "Latitude", "Longitude", "Depth (m)", "Duration (Hrs)"
]
print(df[numeric_cols].describe().round(3))

# Save a clean copy of the original records for analysis.
# This does not overwrite or modify the raw Excel file.
df.to_csv(
    report_folder / "eda_dataset_snapshot.csv",
    index=False
)

# Plot class distribution
df[target].value_counts().sort_index().plot(
    kind="bar", rot=0
)
plt.title("Pufferfish Presence vs Absence")
plt.xlabel("Pufferfish presence (0 = absent, 1 = present)")
plt.ylabel("Number of haul records")
plt.tight_layout()
plt.savefig(
    report_folder / "target_distribution.png",
    dpi=200
)
plt.close()

# Plot depth by presence label
df.boxplot(column="Depth (m)", by=target)
plt.title("Depth by Pufferfish Presence")
plt.suptitle("")
plt.xlabel("Pufferfish presence (0 = absent, 1 = present)")
plt.ylabel("Depth (m)")
plt.tight_layout()
plt.savefig(
    report_folder / "depth_by_presence.png",
    dpi=200
)
plt.close()

print("\nSaved analysis files to:", report_folder)
print("Exploratory data analysis complete.")
