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
from pufferfish.features.build import add_time_features, add_gear_features, add_recent_sightings, recent_positive_counts
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
        df = add_recent_sightings(add_gear_features(add_time_features(obs.sort_values("date").reset_index(drop=True))))
        for c in ("sst_prev7", "sst_trend14"):
            if c not in df:
                df[c] = np.nan
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
        for c in ("sst_prev7", "sst_trend14"):
            if c in hauls:
                obs[c] = hauls[c].values
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

    def _scenario_rows(self, dates, lat, lon, depth, gear, sst_prev7, recent_pos, bundle, obs):
        """Feature rows for 'what if' questions. Inputs the user did not give take a typical value from the training
        data; the recent-sightings count is read from the recorded hauls unless the user supplies it."""
        n = len(dates)
        row = pd.DataFrame({"date": pd.to_datetime(list(dates)), "lat": [lat] * n, "lon": [lon] * n,
                            "depth": [depth] * n, "gear": [gear] * n})
        row = add_gear_features(add_time_features(row))
        row["sst_prev7"] = np.nan if sst_prev7 is None else sst_prev7
        row["sst_trend14"] = np.nan
        rp = (recent_positive_counts(row["date"], row["lat"], row["lon"], obs) if recent_pos is None
              else np.full(n, float(recent_pos)))
        row["recent_pos"], row["recent_pos_log"] = rp, np.log1p(rp)
        for c in bundle["features"]:
            if row[c].isna().any():
                row[c] = row[c].fillna(bundle["medians"].get(c, 0.0))
        return row

    def _season_counts(self, obs, dates):
        doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
        od = obs["date"].dt.dayofyear.to_numpy()
        gap = np.abs(((od[None, :] - doy[:, None] + 182) % 365) - 182)
        return (gap <= 30).sum(1)

    def predict(self, date, lat, lon, depth, gear="Ring seine", sst_prev7=None, n_boot=200, seed=0, recent_pos=None):
        """Probability of pufferfish in a haul with these conditions, with a 90% bootstrap range.

        Returns a dict. status is 'insufficient_data' (no probability) when the position is farther than
        coverage_km from every observed haul, or when fewer than 10 observed hauls were taken within +-30 days
        of the same time of year.
        """
        bundle = self._bundle()
        obs = self._features(self.observations())
        dist, n_near = self.coverage(lat, lon)
        row = self._scenario_rows([date], lat, lon, depth, gear, sst_prev7, recent_pos, bundle, obs)
        cols = bundle["features"]
        n_season = int(self._season_counts(obs, row["date"])[0])
        out = {"nearest_haul_km": round(dist, 1), "hauls_within_coverage": n_near, "hauls_same_season": n_season,
               "model": bundle["name"], "features_used": cols, "base_rate": float(obs["puffer"].mean()),
               "recent_pos_used": float(row["recent_pos"].iloc[0])}
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

    def risk_curve(self, start, weeks=12, step_days=7, lat=None, lon=None, depth=None, gear="Ring seine",
                   n_boot=100, seed=0):
        """Chance of pufferfish for a run of dates at one place (a 'risk calendar'), with a 90% bootstrap band.
        Dates where the twin has too little data get no number (status 'insufficient_data')."""
        bundle = self._bundle()
        obs = self._features(self.observations())
        lat = float(obs["lat"].median()) if lat is None else lat
        lon = float(obs["lon"].median()) if lon is None else lon
        depth = float(obs["depth"].median()) if depth is None else depth
        dates = pd.date_range(pd.Timestamp(start), periods=int(weeks), freq=f"{int(step_days)}D")
        rows = self._scenario_rows(dates, lat, lon, depth, gear, None, None, bundle, obs)
        cols = bundle["features"]
        p = bundle["model"].predict_proba(rows[cols])[:, 1]
        rng, y, boots = np.random.default_rng(seed), obs["puffer"].to_numpy(), []
        for _ in range(n_boot):
            i = rng.integers(0, len(obs), len(obs))
            if y[i].sum() == 0:
                continue
            boots.append(clone(bundle["model"]).fit(obs.iloc[i][cols], y[i]).predict_proba(rows[cols])[:, 1])
        lo, hi = np.percentile(np.array(boots), [5, 95], axis=0)
        dist, _ = self.coverage(lat, lon)
        n_season = self._season_counts(obs, dates)
        ok = (dist <= CFG["twin"]["coverage_km"]) & (n_season >= 10)
        return pd.DataFrame({"date": dates, "p": np.where(ok, p, np.nan), "p_low": np.where(ok, lo, np.nan),
                             "p_high": np.where(ok, hi, np.nan), "hauls_same_season": n_season,
                             "status": np.where(ok, "ok", "insufficient_data")})

    def explain(self) -> pd.DataFrame:
        """What the current model relies on. For a logistic model: the weight of each (standardised) input.
        Season enters as cosine/sine pairs, so read those together as one smooth curve through the year."""
        bundle = self._bundle()
        m = bundle["model"].named_steps["m"] if hasattr(bundle["model"], "named_steps") else bundle["model"]
        cols = bundle["features"]
        if hasattr(m, "coef_"):
            w = m.coef_[0]
            return pd.DataFrame({"input": cols, "weight": w, "effect_on_odds_per_1_sd": np.exp(w), "kind": "logistic weight"})
        if hasattr(m, "feature_importances_"):
            return pd.DataFrame({"input": cols, "weight": m.feature_importances_, "effect_on_odds_per_1_sd": np.nan,
                                 "kind": "forest importance"})
        return pd.DataFrame({"input": cols, "weight": np.nan, "effect_on_odds_per_1_sd": np.nan, "kind": "not available"})

    def drift(self, window=None) -> dict:
        """Is the model still working on the newest hauls? Refit on everything before the last `window` hauls,
        predict those hauls, and compare the squared error with simply guessing the earlier base rate."""
        window = int(window or CFG["twin"].get("drift_window", 30))
        meta = self.meta()
        name = meta.get("model") or CFG["twin"]["default_model"]
        obs = self._features(self.observations())
        if len(obs) < window + 60:
            return {"status": "not_enough_data", "message": f"needs at least {window + 60} hauls", "window": window}
        model, cols = candidate(name)
        y = obs["puffer"].to_numpy()
        tr, te = np.arange(len(obs) - window), np.arange(len(obs) - window, len(obs))
        p = clone(model).fit(obs.iloc[tr][cols], y[tr]).predict_proba(obs.iloc[te][cols])[:, 1]
        base = float(y[tr].mean())
        brier, ref = float(np.mean((p - y[te]) ** 2)), float(np.mean((base - y[te]) ** 2))
        skill = 1 - brier / ref if ref > 0 else float("nan")
        status = "ok" if (np.isnan(skill) or skill > 0) else "watch"
        msg = ("The model predicts the newest hauls better than a constant guess." if status == "ok"
               else "The model does worse than a constant guess on the newest hauls: review the data and consider retraining.")
        return {"status": status, "window": window, "model": name, "newest_positives": int(y[te].sum()),
                "brier": round(brier, 4), "brier_constant_guess": round(ref, 4),
                "brier_skill": None if np.isnan(skill) else round(skill, 3),
                "from": str(obs["date"].iloc[te[0]].date()), "to": str(obs["date"].iloc[te[-1]].date()), "message": msg}