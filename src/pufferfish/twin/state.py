"""The 'digital twin' state: what the system currently believes, how it got there, and what it refuses to claim.

A twin here means four concrete, testable things:
  1. STATE     - accepted observations + the current fitted model, saved on disk.
  2. UPDATE    - new hauls are submitted, reviewed by a person, accepted or rejected; retraining is explicit.
  3. HISTORY   - every version is recorded (date, data hash, model, out-of-time metrics), so changes are traceable.
  4. SCENARIOS - ask 'what is the pufferfish chance for this date / depth / temperature?' with an uncertainty range,
                 and get 'insufficient data' instead of a number where the data cannot support one.
It is NOT a physical simulation of the sea, and it does not forecast outside the observed fishing ground.
"""
from __future__ import annotations
import json, uuid
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from pufferfish.config import CFG, PROJECT_ROOT, path
from pufferfish.environmental.sst import haversine_km
from pufferfish.features.build import add_time_features, add_gear_features
from pufferfish.models import evaluate as ev
from pufferfish.models.estimators import candidate
from pufferfish.models.train import data_hash

GEARS = ["Ring seine", "Mini ring seine", "Large ring seine"]
OBS_COLS = ["date", "lat", "lon", "depth", "gear", "puffer", "source"]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TwinState:
    def __init__(self, state_dir: Path | None = None):
        self.dir = Path(state_dir) if state_dir else path(CFG["twin"]["state_dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.obs_file, self.pending_file = self.dir / "observations.csv", self.dir / "pending.json"
        self.history_file, self.model_file = self.dir / "history.jsonl", self.dir / "model.joblib"

    # ---------- state ----------
    @property
    def initialised(self):
        return self.obs_file.exists() and self.model_file.exists()

    def observations(self) -> pd.DataFrame:
        return pd.read_csv(self.obs_file, parse_dates=["date"])

    def history(self) -> pd.DataFrame:
        if not self.history_file.exists():
            return pd.DataFrame()
        return pd.DataFrame([json.loads(l) for l in self.history_file.read_text().splitlines() if l.strip()])

    def _log(self, **rec):
        h = self.history()
        rec = {"version": len(h) + 1, "timestamp_utc": _now(), **rec}
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def meta(self) -> dict:
        """Latest MODEL version (build or retrain); review events carry no model and are skipped."""
        h = self.history()
        if len(h) and "model" in h:
            h = h[h["model"].notna()]
        return h.iloc[-1].to_dict() if len(h) else {}

    def _bundle(self) -> dict:
        """Load the saved model. If it was saved with another scikit-learn version and cannot predict here
        (common after deploying to a server), refit the same model from the saved hauls instead of failing."""
        b = joblib.load(self.model_file)
        try:
            b["model"].predict_proba(pd.DataFrame([b["medians"]])[b["features"]])
        except Exception:
            df = self._features(self.observations())
            model, cols = candidate(b["name"])
            b = {"model": clone(model).fit(df[cols], df["puffer"].to_numpy()), "name": b["name"], "features": cols,
                 "medians": df[cols].median().to_dict()}
            try:
                joblib.dump(b, self.model_file)
            except OSError:
                pass
        return b

    # ---------- build / rebuild ----------
    def _features(self, obs: pd.DataFrame) -> pd.DataFrame:
        df = add_gear_features(add_time_features(obs.sort_values("date").reset_index(drop=True)))
        if "sst_prev7" not in df:
            df["sst_prev7"] = np.nan
        return df

    def _fit(self, obs, model_name, event, note=""):
        df = self._features(obs)
        model, cols = candidate(model_name)
        if df[cols].isna().all().any():
            raise ValueError(f"Model {model_name} needs features that are missing: "
                             f"{[c for c in cols if df[c].isna().all()]}")
        y = df["puffer"].to_numpy()
        fitted = clone(model).fit(df[cols], y)
        idx, p = ev.chrono_predictions(df, y, model, cols)
        s, (lo, hi) = ev.score(y[idx], p), ev.bootstrap_ci(y[idx], p)
        bmodel, bcols = candidate("seasonal_baseline")
        bidx, bp = ev.chrono_predictions(df, y, bmodel, bcols)
        joblib.dump({"model": fitted, "name": model_name, "features": cols,
                     "medians": df[cols].median().to_dict()}, self.model_file)
        obs.to_csv(self.obs_file, index=False)
        return self._log(event=event, note=note, model=model_name, features=cols, n_hauls=int(len(df)),
                         n_positive=int(y.sum()), data_hash=data_hash(df[["date", "lat", "lon", "depth", "puffer"]]),
                         out_of_time_ap=round(s["average_precision"], 4), out_of_time_ap_ci=[round(lo, 4), round(hi, 4)],
                         out_of_time_roc_auc=round(s["roc_auc"], 4), seasonal_baseline_ap=round(ev.score(y[bidx], bp)["average_precision"], 4),
                         base_rate=round(s["prevalence"], 4))

    def initialise(self, hauls: pd.DataFrame, model_name=None, source="CMFRI 2023-26"):
        obs = hauls[["date", "lat", "lon", "depth", "gear", "puffer"]].copy()
        obs["source"] = source
        if "sst_prev7" in hauls:
            obs["sst_prev7"] = hauls["sst_prev7"].values
        self.pending_file.write_text("[]")
        return self._fit(obs, model_name or CFG["twin"]["default_model"], "initial", "built from the supplied CMFRI hauls")

    def retrain(self, model_name=None, note="manual retrain"):
        return self._fit(self.observations(), model_name or self.meta().get("model") or CFG["twin"]["default_model"],
                         "retrain", note)

    # ---------- new observations: submit -> review -> accept ----------
    def pending(self) -> list[dict]:
        return json.loads(self.pending_file.read_text()) if self.pending_file.exists() else []

    def validate(self, rec: dict) -> list[str]:
        errs, obs = [], self.observations()
        try:
            d = pd.Timestamp(rec["date"])
            if d > pd.Timestamp(datetime.now(timezone.utc).date()):
                errs.append("date is in the future")
        except Exception:
            errs.append("date is not valid"); d = None
        pad = 0.5
        if not (obs["lat"].min() - pad <= float(rec.get("lat", np.nan)) <= obs["lat"].max() + pad):
            errs.append("latitude is outside the observed fishing ground (+0.5 degree)")
        if not (obs["lon"].min() - pad <= float(rec.get("lon", np.nan)) <= obs["lon"].max() + pad):
            errs.append("longitude is outside the observed fishing ground (+0.5 degree)")
        if not (1 <= float(rec.get("depth", 0)) <= 100):
            errs.append("depth must be between 1 and 100 m")
        if rec.get("gear") not in GEARS:
            errs.append(f"gear must be one of {GEARS}")
        if int(rec.get("puffer", -1)) not in (0, 1):
            errs.append("puffer must be 0 (confirmed absent) or 1 (present)")
        if d is not None:
            dup = obs[(obs["date"] == d) & ((obs["lat"] - float(rec["lat"])).abs() < 1e-3) & ((obs["lon"] - float(rec["lon"])).abs() < 1e-3)]
            if len(dup):
                errs.append("a haul with this date and position already exists")
        return errs

    def submit(self, rec: dict, submitted_by="anonymous"):
        errs = self.validate(rec)
        if errs:
            return False, errs, None
        item = {"id": uuid.uuid4().hex[:8], "status": "pending", "submitted_utc": _now(), "submitted_by": submitted_by,
                "record": {k: rec[k] for k in ("date", "lat", "lon", "depth", "gear", "puffer")}}
        self.pending_file.write_text(json.dumps(self.pending() + [item], indent=2, default=str))
        return True, [], item["id"]

    def review(self, item_id: str, accept: bool, reviewer: str, note=""):
        items, hit = self.pending(), None
        for it in items:
            if it["id"] == item_id and it["status"] == "pending":
                hit = it
        if hit is None:
            raise KeyError(f"no pending submission {item_id}")
        hit.update(status="accepted" if accept else "rejected", reviewed_by=reviewer, reviewed_utc=_now(), note=note)
        if accept:
            obs = self.observations()
            r = hit["record"]
            row = {"date": pd.Timestamp(r["date"]), "lat": float(r["lat"]), "lon": float(r["lon"]), "depth": float(r["depth"]),
                   "gear": r["gear"], "puffer": int(r["puffer"]), "source": f"reviewed submission {item_id}"}
            pd.concat([obs, pd.DataFrame([row])], ignore_index=True).to_csv(self.obs_file, index=False)
            self._log(event="observation_accepted", note=f"{item_id} accepted by {reviewer}; retrain to use it")
        self.pending_file.write_text(json.dumps(items, indent=2, default=str))

    # ---------- scenarios ----------
    def coverage(self, lat, lon):
        obs = self.observations()
        d = haversine_km(lat, lon, obs["lat"].to_numpy(), obs["lon"].to_numpy())
        return float(d.min()), int((d <= CFG["twin"]["coverage_km"]).sum())

    def predict(self, date, lat, lon, depth, gear="Ring seine", sst_prev7=None, n_boot=200, seed=0):
        """Probability of pufferfish in a haul with these conditions, with a 90% bootstrap range.

        Returns a dict. status is 'insufficient_data' (no probability) when the position is farther than
        coverage_km from every observed haul, or when fewer than 10 observed hauls were taken within +-30 days
        of the same time of year.
        """
        bundle = self._bundle()
        obs = self._features(self.observations())
        dist, n_near = self.coverage(lat, lon)
        row = add_gear_features(add_time_features(pd.DataFrame([{"date": pd.Timestamp(date), "lat": lat, "lon": lon,
                                                                 "depth": depth, "gear": gear}])))
        row["sst_prev7"] = np.nan if sst_prev7 is None else sst_prev7
        cols = bundle["features"]
        for c in cols:                                  # unspecified inputs -> typical value from training data
            if pd.isna(row[c].iloc[0]):
                row[c] = bundle["medians"].get(c, 0.0)
        doy_gap = np.abs(((obs["date"].dt.day_of_year - row["date"].dt.day_of_year.iloc[0] + 182) % 365) - 182)
        n_season = int((doy_gap <= 30).sum())
        out = {"nearest_haul_km": round(dist, 1), "hauls_within_coverage": n_near, "hauls_same_season": n_season,
               "model": bundle["name"], "features_used": cols, "base_rate": float(obs["puffer"].mean())}
        if dist > CFG["twin"]["coverage_km"] or n_season < 10:
            reason = (f"nearest observed haul is {dist:.0f} km away (limit {CFG['twin']['coverage_km']} km)"
                      if dist > CFG["twin"]["coverage_km"] else f"only {n_season} observed hauls within 30 days of this time of year")
            return {**out, "status": "insufficient_data", "reason": reason, "p": None, "p_low": None, "p_high": None}
        p = float(bundle["model"].predict_proba(row[cols])[:, 1][0])
        rng, y, ps = np.random.default_rng(seed), obs["puffer"].to_numpy(), []
        for _ in range(n_boot):
            i = rng.integers(0, len(obs), len(obs))
            if y[i].sum() == 0:
                continue
            ps.append(clone(bundle["model"]).fit(obs.iloc[i][cols], y[i]).predict_proba(row[cols])[:, 1][0])
        lo, hi = np.percentile(ps, [5, 95])
        return {**out, "status": "ok", "reason": "", "p": p, "p_low": float(lo), "p_high": float(hi)}