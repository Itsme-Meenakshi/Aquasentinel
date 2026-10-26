"""Load the cleaned haul table (output of notebooks/05_data_cleaning.py) with tidy column names."""
import pandas as pd
from pufferfish.config import path, TARGET_RAW, LEAKAGE_COLUMNS

RENAME = {
    "Date": "date", "Season": "season", "Latitude": "lat", "Longitude": "lon",
    "Depth (m)": "depth", "Gear": "gear", TARGET_RAW: "puffer",
}


def normalise_gear(series: pd.Series) -> pd.Series:
    """Collapse spelling variants into three gear classes."""
    g = series.astype(str).str.strip().str.lower()
    return g.map(lambda s: "Mini ring seine" if "mini" in s else ("Large ring seine" if "large" in s else "Ring seine"))


def load_hauls(cleaned_csv=None) -> pd.DataFrame:
    """Return hauls sorted by date, with pre-haul columns only plus the target.

    Post-haul columns (catch, CPUE, duration, species) are DROPPED here so they can never
    leak into a model by accident.
    """
    cleaned_csv = cleaned_csv or path("data.cleaned_file")
    if not cleaned_csv.exists():
        raise FileNotFoundError(
            f"{cleaned_csv} not found. Put the Excel file in data/raw/ and run "
            "python notebooks/05_data_cleaning.py first."
        )
    df = pd.read_csv(cleaned_csv)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    df = df.rename(columns=RENAME)
    df["date"] = pd.to_datetime(df["date"])
    df["gear"] = normalise_gear(df["gear"])
    df = df[["date", "season", "lat", "lon", "depth", "gear", "puffer"]]
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    assert set(df["puffer"].unique()) <= {0, 1}, "target must be 0/1"
    return df
