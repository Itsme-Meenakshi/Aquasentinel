"""Feature construction. Every feature must be knowable BEFORE the haul."""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical day-of-year features (December and January are neighbours, not opposites)."""
    out = df.copy()
    angle = 2 * np.pi * out["date"].dt.dayofyear / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(angle), np.cos(angle)
    out["doy_sin2"], out["doy_cos2"] = np.sin(2 * angle), np.cos(2 * angle)  # second harmonic: two peaks per year
    out["month"] = out["date"].dt.month
    return out


def add_gear_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gear_mini"] = (out["gear"] == "Mini ring seine").astype(int)
    return out


def build_features(hauls: pd.DataFrame, sst_features: pd.DataFrame | None = None) -> pd.DataFrame:
    """hauls: output of data.loader.load_hauls. sst_features: output of environmental.sst.match_hauls_to_sst."""
    out = add_gear_features(add_time_features(hauls))
    if sst_features is not None:
        assert len(sst_features) == len(out)
        for c in sst_features.columns:
            out[c] = sst_features[c].values
    else:
        out["sst_prev7"] = np.nan
    return out
