"""Honest evaluation for small data.

1. Chronological (expanding window): train on earlier hauls, test on the NEXT block. This mimics real use
   (predicting the future) and is the number to quote.
2. Repeated stratified k-fold: shows how much the answer depends on which hauls land in the test set.
   It is optimistic (neighbouring days look alike), so it is a stress test, never the headline.
Predictions from all test blocks are pooled before scoring (a single block holds only ~8 positives).
Uncertainty: bootstrap confidence intervals. Significance: label-permutation test.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from pufferfish.config import CFG
from pufferfish.models.estimators import candidate

EV = CFG["evaluation"]


def chrono_splits(n, folds=None, min_train_fraction=None):
    folds, mtf = folds or EV["chrono_folds"], min_train_fraction or EV["min_train_fraction"]
    edges = np.linspace(int(n * mtf), n, folds + 1).astype(int)
    return [(np.arange(0, a), np.arange(a, b)) for a, b in zip(edges[:-1], edges[1:])]


def score(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    prev = y.mean()
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.full_like(p, prev))
    return {"average_precision": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "brier": brier, "brier_skill": 1 - brier / brier_ref, "log_loss": log_loss(y, p),
            "prevalence": prev}


def chrono_predictions(df, y, model, cols, splits=None):
    """Out-of-time predictions pooled over all test blocks. Returns (index array, probabilities)."""
    idx, pr = [], []
    for tr, te in (splits or chrono_splits(len(df))):
        m = clone(model).fit(df.iloc[tr][cols], y[tr])
        idx.append(te)
        pr.append(m.predict_proba(df.iloc[te][cols])[:, 1])
    return np.concatenate(idx), np.concatenate(pr)


def bootstrap_ci(y, p, n=None, seed=None, metric=average_precision_score):
    n, rng = n or EV["bootstrap"], np.random.default_rng(seed if seed is not None else EV["random_state"])
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        vals.append(metric(y[i], p[i]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def repeated_cv_ap(df, y, model, cols, repeats=None, seed=None):
    """Mean +- sd of pooled average precision over repeats of stratified 5-fold CV."""
    rep = repeats or EV["cv_repeats"]
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=rep, random_state=seed or EV["random_state"])
    aps, buf, k = [], np.zeros(len(df)), 0
    for tr, te in rskf.split(df, y):
        buf[te] = clone(model).fit(df.iloc[tr][cols], y[tr]).predict_proba(df.iloc[te][cols])[:, 1]
        k += 1
        if k % 5 == 0:
            aps.append(average_precision_score(y, buf)); buf = np.zeros(len(df))
    return float(np.mean(aps)), float(np.std(aps))


def permutation_pvalue(df, y, model, cols, observed_ap, n=None, seed=None):
    """P(AP >= observed) if the features carried no information about the label (labels shuffled)."""
    n, rng = n or EV["permutations"], np.random.default_rng(seed if seed is not None else EV["random_state"])
    splits, null = chrono_splits(len(df)), []
    for _ in range(n):
        yp = rng.permutation(y)
        # score against the permuted labels so train and test labels are both shuffled
        _, pr = chrono_predictions(df, yp, model, cols, splits)
        idx = np.concatenate([te for _, te in splits])
        null.append(average_precision_score(yp[idx], pr))
    return (1 + sum(v >= observed_ap for v in null)) / (n + 1), float(np.mean(null))


def compare_candidates(df, names=None, with_cv=True, with_perm=False, perm_for=()):
    y = df["puffer"].to_numpy()
    splits = chrono_splits(len(df))
    test_idx = np.concatenate([te for _, te in splits])
    rows = []
    for name in (names or list(__import__("pufferfish.models.estimators", fromlist=["CANDIDATES"]).CANDIDATES)):
        model, cols = candidate(name)
        if df[cols].isna().all().any():       # e.g. SST not available yet
            rows.append({"model": name, "status": "skipped (feature missing)"}); continue
        idx, p = chrono_predictions(df, y, model, cols, splits)
        yt = y[idx]
        s = score(yt, p)
        if name == "prevalence_baseline":
            s["roc_auc"] = 0.5   # a constant predictor has no ranking ability; pooled folds with different constants would give a meaningless value
        lo, hi = bootstrap_ci(yt, p)
        row = {"model": name, "n_predictors": len(cols), "n_test_hauls": len(idx), "test_positives": int(yt.sum()),
               **{f"chrono_{k}": v for k, v in s.items()}, "chrono_ap_ci_low": lo, "chrono_ap_ci_high": hi,
               "status": "ok"}
        if with_cv:
            m, sd = repeated_cv_ap(df, y, model, cols)
            row.update(cv_ap_mean=m, cv_ap_sd=sd)
        if with_perm and name in perm_for:
            pv, nullm = permutation_pvalue(df, y, model, cols, s["average_precision"])
            row.update(perm_p_value=pv, perm_null_ap_mean=nullm)
        rows.append(row)
    return pd.DataFrame(rows)


def learning_curve(df, names=("logistic_season2", "seasonal_baseline"), fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
                   repeats=30, test_fraction=0.4, seed=None):
    """How does out-of-time skill change with the amount of real training data?

    The LAST test_fraction of hauls (by date) is held out once. The earlier hauls form a pool; for each fraction
    we train on random subsets of that pool (repeated) and score on the fixed held-out block. Only real hauls are used.
    """
    rng = np.random.default_rng(seed if seed is not None else EV["random_state"])
    y = df["puffer"].to_numpy()
    cut = int(len(df) * (1 - test_fraction))
    pool, test = np.arange(cut), np.arange(cut, len(df))
    rows = []
    for name in names:
        model, cols = candidate(name)
        for f in fractions:
            m = max(int(round(len(pool) * f)), 10)
            aps = []
            for _ in range(repeats if f < 1 else 1):
                tr = np.sort(rng.choice(pool, m, replace=False)) if f < 1 else pool
                if y[tr].sum() < 2 or y[tr].sum() == len(tr):
                    continue
                p = clone(model).fit(df.iloc[tr][cols], y[tr]).predict_proba(df.iloc[test][cols])[:, 1]
                aps.append(average_precision_score(y[test], p))
            if aps:
                rows.append({"model": name, "train_hauls": m, "train_positives_mean": float(y[pool].mean() * m),
                             "ap_mean": float(np.mean(aps)), "ap_sd": float(np.std(aps)),
                             "test_hauls": len(test), "test_positives": int(y[test].sum()),
                             "no_skill_ap": float(y[test].mean())})
    return pd.DataFrame(rows)
