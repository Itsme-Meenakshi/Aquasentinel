
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
raw_folder = PROJECT_ROOT / "data" / "raw"
report_folder = PROJECT_ROOT / "reports"
report_folder.mkdir(exist_ok=True)

files = list(raw_folder.glob("*.xlsx"))
if not files:
    raise FileNotFoundError("No Excel file found in data/raw/")

df = pd.read_excel(files[0])
df.columns = df.columns.str.strip()

target = "PUFFER Presence (1/0)"
df["Date"] = pd.to_datetime(
    df["Date"], format="%d.%m.%Y", errors="coerce"
)

# 1. Seasonal counts and presence rates
season_summary = df.groupby("Season").agg(
    Total_Hauls=(target, "size"),
    Puffer_Present=(target, "sum"),
    Presence_Rate=(target, "mean")
)
season_summary["Presence_Rate"] *= 100

print("\n--- SEASONAL SUMMARY ---")
print(season_summary.round(2))
season_summary.to_csv(report_folder / "season_summary.csv")

# 2. Monthly counts and presence rates
df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

monthly_summary = df.groupby("YearMonth").agg(
    Total_Hauls=(target, "size"),
    Puffer_Present=(target, "sum"),
    Presence_Rate=(target, "mean")
)
monthly_summary["Presence_Rate"] *= 100

print("\n--- MONTHLY SUMMARY ---")
print(monthly_summary)
monthly_summary.to_csv(report_folder / "monthly_summary.csv")

# Plot monthly haul counts
monthly_summary["Total_Hauls"].plot(kind="bar", figsize=(12, 5))
plt.title("Monthly Haul Records")
plt.xlabel("Year and month")
plt.ylabel("Number of hauls")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(report_folder / "monthly_haul_counts.png", dpi=200)
plt.close()

# Plot monthly pufferfish presence rate
monthly_summary["Presence_Rate"].plot(
    kind="line", marker="o", figsize=(12, 5)
)
plt.title("Monthly Pufferfish Presence Rate")
plt.xlabel("Year and month")
plt.ylabel("Presence rate (%)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(report_folder / "monthly_presence_rate.png", dpi=200)
plt.close()

# 3. Map haul locations by presence label
fig, ax = plt.subplots(figsize=(8, 7))

for label, name in [(0, "Absent"), (1, "Present")]:
    subset = df[df[target] == label]
    ax.scatter(
        subset["Longitude"],
        subset["Latitude"],
        label=name,
        alpha=0.75,
        s=35
    )

ax.set_title("Recorded Haul Locations by Pufferfish Presence")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(report_folder / "haul_location_map.png", dpi=200)
plt.close()

print("\nSaved files to:", report_folder)
print("Spatial and temporal analysis complete.")
