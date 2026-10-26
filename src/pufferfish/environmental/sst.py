"""Sea-surface temperature (NOAA OISST v2.1) download and haul matching.

Timing rule: features use SST from the days BEFORE the haul only (window ends the day before),
so the same feature could be computed in real time before going fishing.
Matching rule: use the nearest ocean pixel (the OISST grid is 0.25 degrees, so pixels next to the
coast are masked as land) within `max_match_km` that has enough valid days. The distance used is
saved for every haul, so matching quality is documented, not hidden.
"""
from __future__ import annotations
import io
import time
import numpy as np
import pandas as pd
from pufferfish.config import CFG, path

EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


# ---------- download ----------
def erddap_url(start, end, bbox, base=None, dataset=None) -> str:
    base, dataset = base or CFG["sst"]["erddap_base"], dataset or CFG["sst"]["dataset_id"]
    t0, t1 = pd.Timestamp(start).strftime("%Y-%m-%dT12:00:00Z"), pd.Timestamp(end).strftime("%Y-%m-%dT12:00:00Z")
    return (f"{base}/{dataset}.csv?sst[({t0}):1:({t1})][(0.0):1:(0.0)]"
            f"[({bbox['lat_min']}):1:({bbox['lat_max']})][({bbox['lon_min']}):1:({bbox['lon_max']})]")


def parse_erddap_csv(text: str) -> pd.DataFrame:
    """ERDDAP CSV has a units row under the header; drop it. Returns columns date, lat, lon, sst."""
    df = pd.read_csv(io.StringIO(text), skiprows=[1], na_values=["NaN", ""])
    df = df.rename(columns={"time": "date", "latitude": "lat", "longitude": "lon"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df[["date", "lat", "lon", "sst"]]


def manual_download_url(start, end, bbox=None) -> str:
    """One URL covering the whole period. Paste it into a browser: it downloads a CSV for --sst-csv."""
    bbox = bbox or CFG["sst"]["bbox"]
    return erddap_url(start, end, bbox).replace("[", "%5B").replace("]", "%5D")


def _get_with_retries(getter, a, b, bbox, attempts=3, wait=3.0):
    """Try each configured ERDDAP server, retrying each a few times (these servers are often slow)."""
    bases = CFG["sst"].get("erddap_bases") or [CFG["sst"]["erddap_base"]]
    last = None
    for base in bases:
        for k in range(attempts):
            try:
                r = getter(erddap_url(a, b, bbox, base=base))
                if r.status_code == 200:
                    return r
                last = RuntimeError(f"ERDDAP request failed ({r.status_code}) at {base}: {r.text[:200]}")
            except Exception as e:   # timeout, DNS, blocked network ...
                last = e
            if k < attempts - 1:
                time.sleep(wait)
    raise last


def fetch_oisst(start, end, bbox=None, cache_file=None, refresh=False, getter=None) -> pd.DataFrame:
    """Download daily SST for the bounding box (one request per year) and cache it as CSV."""
    import requests
    getter = getter or (lambda url: requests.get(url, timeout=(30, 300)))
    bbox, cache_file = bbox or CFG["sst"]["bbox"], cache_file or path("sst.cache_file")
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if cache_file.exists() and not refresh:
        cached = pd.read_csv(cache_file, parse_dates=["date"])
        if cached["date"].min() <= start and cached["date"].max() >= end - pd.Timedelta(days=3):
            return cached
    parts = []
    for y0 in pd.date_range(start, end, freq="YS-JAN").union([start]):
        a, b = max(y0, start), min(pd.Timestamp(year=y0.year, month=12, day=31), end)
        if a > b:
            continue
        parts.append(parse_erddap_csv(_get_with_retries(getter, a, b, bbox).text))
    grid = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "lat", "lon"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(cache_file, index=False)
    return grid


def load_manual_csv(csv_path) -> pd.DataFrame:
    """Use a manually downloaded grid (columns time/date, latitude/lat, longitude/lon, sst)."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"time": "date", "latitude": "lat", "longitude": "lon"})
    if str(df.iloc[0, 0]).lower() in ("utc",):  # units row
        df = df.iloc[1:]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    for c in ("lat", "lon", "sst"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "lat", "lon", "sst"]].dropna(subset=["date"]).reset_index(drop=True)


# ---------- matching ----------
def match_hauls_to_sst(hauls: pd.DataFrame, grid: pd.DataFrame, window_days=None, min_valid_days=None,
                       max_match_km=None) -> pd.DataFrame:
    """For each haul return pre-haul SST features and matching-quality columns (same row order as hauls)."""
    window_days = window_days or CFG["sst"]["window_days"]
    min_valid_days = min_valid_days or CFG["sst"]["min_valid_days"]
    max_match_km = max_match_km or CFG["sst"]["max_match_km"]

    series = {}
    for (la, lo), g in grid.dropna(subset=["sst"]).groupby(["lat", "lon"]):
        series[(la, lo)] = g.set_index("date")["sst"].sort_index()
    pix = np.array(list(series.keys())) if series else np.empty((0, 2))

    rows = []
    for _, h in hauls.iterrows():
        d = pd.Timestamp(h["date"]).normalize()
        rec = dict(sst_prev7=np.nan, sst_trend14=np.nan, sst_match_km=np.nan, sst_pixel_lat=np.nan,
                   sst_pixel_lon=np.nan, sst_valid_days=0, sst_status="no_grid")
        if len(pix):
            dist = haversine_km(h["lat"], h["lon"], pix[:, 0], pix[:, 1])
            rec["sst_status"] = "no_pixel_within_limit"
            for i in np.argsort(dist):
                if dist[i] > max_match_km:
                    break
                s = series[tuple(pix[i])]
                recent = s.loc[d - pd.Timedelta(days=window_days): d - pd.Timedelta(days=1)]
                if len(recent) < min_valid_days:
                    rec["sst_status"] = "insufficient_days"
                    continue
                before = s.loc[d - pd.Timedelta(days=2 * window_days): d - pd.Timedelta(days=window_days + 1)]
                rec.update(sst_prev7=recent.mean(), sst_match_km=float(dist[i]), sst_pixel_lat=pix[i][0],
                           sst_pixel_lon=pix[i][1], sst_valid_days=len(recent), sst_status="matched",
                           sst_trend14=(recent.mean() - before.mean()) if len(before) >= min_valid_days else np.nan)
                break
        rows.append(rec)
    return pd.DataFrame(rows, index=hauls.index)


def matching_report(sst_features: pd.DataFrame) -> dict:
    n = len(sst_features)
    ok = sst_features["sst_status"] == "matched"
    return {"hauls": n, "matched": int(ok.sum()), "matched_pct": round(100 * ok.mean(), 1),
            "median_match_km": float(sst_features.loc[ok, "sst_match_km"].median()) if ok.any() else None,
            "max_match_km": float(sst_features.loc[ok, "sst_match_km"].max()) if ok.any() else None,
            "unmatched_by_reason": sst_features.loc[~ok, "sst_status"].value_counts().to_dict()}
