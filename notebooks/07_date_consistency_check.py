
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
files = list((ROOT / "data" / "raw").glob("*.xlsx"))

if not files:
    raise FileNotFoundError("No Excel file found in data/raw/")

df = pd.read_excel(files[0])
df.columns = df.columns.str.strip()

parsed_dates = pd.to_datetime(
    df["Date"], format="%d.%m.%Y", errors="coerce"
)

expected_month = parsed_dates.dt.month_name()
actual_month = df["Month"].astype("string").str.strip()

mismatch = (
    expected_month.str.casefold()
    != actual_month.str.casefold()
)

print("Number of mismatched records:", int(mismatch.sum()))

columns = [
    "Date", "Month", "Season",
    "Latitude", "Longitude",
    "PUFFER Presence (1/0)"
]

print("\nRecords requiring review:")
print(df.loc[mismatch, columns].to_string(index=True))
