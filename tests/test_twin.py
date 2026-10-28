import pandas as pd
import pytest
from pufferfish.features.build import build_features
from pufferfish.twin.state import TwinState


@pytest.fixture
def tw(tmp_path, hauls):
    t = TwinState(tmp_path)
    t.initialise(build_features(hauls), "logistic_season2")
    return t


def test_initialise_logs_version_one_with_honest_metrics(tw):
    m = tw.meta()
    assert m["version"] == 1 and m["model"] == "logistic_season2" and 0 < m["out_of_time_ap"] <= 1
    assert "seasonal_baseline_ap" in m and len(m["data_hash"]) == 12


def test_prediction_has_range_and_peak_season_is_riskier(tw):
    w = tw.predict("2024-01-10", 9.85, 76.17, 12)
    s = tw.predict("2024-06-10", 9.85, 76.17, 12)
    assert w["status"] == "ok" and w["p_low"] <= w["p"] <= w["p_high"] and w["p"] > s["p"]


def test_twin_refuses_far_away_positions(tw):
    r = tw.predict("2024-01-10", 11.9, 75.4, 12)
    assert r["status"] == "insufficient_data" and r["p"] is None and "km" in r["reason"]


def test_submission_is_validated(tw):
    ok, errs, _ = tw.submit({"date": "2999-01-01", "lat": 9.85, "lon": 76.17, "depth": 12, "gear": "Ring seine", "puffer": 1})
    assert not ok and any("future" in e for e in errs)
    ok, errs, _ = tw.submit({"date": "2025-05-05", "lat": 15.0, "lon": 76.17, "depth": 12, "gear": "Ring seine", "puffer": 1})
    assert not ok and any("latitude" in e for e in errs)


def test_review_then_retrain_creates_new_version(tw):
    n0 = len(tw.observations())
    ok, _, sid = tw.submit({"date": "2025-05-05", "lat": 9.851, "lon": 76.172, "depth": 12, "gear": "Ring seine", "puffer": 0})
    assert ok and len(tw.observations()) == n0           # pending does NOT change the twin
    tw.review(sid, True, "reviewer")
    assert len(tw.observations()) == n0 + 1
    rec = tw.retrain()
    assert rec["n_hauls"] == n0 + 1 and rec["version"] > 2
    assert {"initial", "observation_accepted", "retrain"} <= set(tw.history().event)


def test_rejected_submission_is_not_added(tw):
    n0 = len(tw.observations())
    _, _, sid = tw.submit({"date": "2025-06-06", "lat": 9.85, "lon": 76.17, "depth": 12, "gear": "Ring seine", "puffer": 1})
    tw.review(sid, False, "reviewer", "bad label")
    assert len(tw.observations()) == n0


class _Broken:                            # mimics the unpickled-with-a-different-version failure
    def predict_proba(self, X):
        raise AttributeError("_fill_dtype")


def test_model_saved_by_another_sklearn_version_is_refitted(tmp_path, hauls):
    import joblib
    from pufferfish.features.build import build_features
    t = TwinState(tmp_path)
    t.initialise(build_features(hauls), "logistic_season2")
    ok = t.predict("2024-01-10", 9.85, 76.17, 12)["p"]
    b = joblib.load(t.model_file)

    b["model"] = _Broken(); joblib.dump(b, t.model_file)
    assert abs(t.predict("2024-01-10", 9.85, 76.17, 12)["p"] - ok) < 1e-9