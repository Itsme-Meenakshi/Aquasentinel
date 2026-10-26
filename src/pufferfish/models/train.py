"""Fit the final model on all accepted hauls and save it with its metadata."""
import json, hashlib
from datetime import datetime, timezone
import joblib
import pandas as pd
from sklearn.base import clone
from pufferfish.config import PROJECT_ROOT
from pufferfish.models.estimators import candidate


def data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]


def fit_final(df, name, out_dir=None):
    model, cols = candidate(name)
    m = clone(model).fit(df[cols], df["puffer"].to_numpy())
    meta = {"model": name, "features": cols, "n_hauls": int(len(df)), "n_positive": int(df["puffer"].sum()),
            "data_hash": data_hash(df[["date", "lat", "lon", "depth", "puffer"]]),
            "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out_dir = out_dir or PROJECT_ROOT / "artifacts" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, out_dir / f"{name}.joblib")
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return m, meta
