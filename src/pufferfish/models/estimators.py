"""Model zoo. Small data (253 hauls, 38 positives) -> few predictors, strong regularisation, honest baselines."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pufferfish.config import FEATURE_SETS


class SeasonalRate(BaseEstimator, ClassifierMixin):
    """Smoothed presence rate by time of year (circular kernel), shrunk toward the overall rate.

    This is the 'dumb but honest' seasonal baseline: any real model must beat it to claim it learned
    more than 'pufferfish are common in some months'. Needs columns doy_sin, doy_cos.
    """
    def __init__(self, kappa=8.0, prior_strength=5.0):
        self.kappa, self.prior_strength = kappa, prior_strength

    def _angle(self, X):
        X = pd.DataFrame(X, columns=getattr(self, "cols_", None))
        return np.arctan2(X["doy_sin"].to_numpy(float), X["doy_cos"].to_numpy(float))

    def fit(self, X, y):
        self.cols_ = list(X.columns)
        self.classes_ = np.array([0, 1])
        self.angle_, self.y_ = self._angle(X), np.asarray(y, float)
        self.base_ = self.y_.mean()
        return self

    def predict_proba(self, X):
        a = self._angle(X)
        w = np.exp(self.kappa * np.cos(a[:, None] - self.angle_[None, :]))
        p = (w @ self.y_ + self.prior_strength * self.base_) / (w.sum(1) + self.prior_strength)
        return np.c_[1 - p, p]


def make_model(kind: str, feature_cols):
    """kind: prevalence | seasonal | logistic | forest."""
    if kind == "prevalence":
        est = DummyClassifier(strategy="prior")
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", est)])
    if kind == "seasonal":
        return SeasonalRate()
    if kind == "logistic":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("m", LogisticRegression(C=0.5, max_iter=2000))])
    if kind == "logistic_balanced":
        # class_weight="balanced" up-weights the rare positive class; ranking (AP) may change, probabilities become over-confident
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("m", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))])
    if kind == "forest":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", RandomForestClassifier(300, max_depth=3, min_samples_leaf=10,
                                                      random_state=0, n_jobs=1))])
    raise ValueError(kind)


# name -> (model kind, feature-set name)
CANDIDATES = {
    "prevalence_baseline": ("prevalence", "season1"),
    "seasonal_baseline": ("seasonal", "season1"),
    "logistic_season1": ("logistic", "season1"),
    "logistic_season2": ("logistic", "season2"),
    "logistic_season2_balanced": ("logistic_balanced", "season2"),
    "logistic_season_depth": ("logistic", "season1_depth"),
    "logistic_season_sst": ("logistic", "season1_sst"),
    "logistic_season_depth_sst": ("logistic", "season1_depth_sst"),
    "forest_prehaul_no_sst": ("forest", "all_prehaul_nosst"),
    "forest_all_prehaul": ("forest", "all_prehaul"),
}


def candidate(name):
    kind, fs = CANDIDATES[name]
    cols = FEATURE_SETS[fs]
    return make_model(kind, cols), cols
