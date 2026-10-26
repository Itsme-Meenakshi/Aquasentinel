import numpy as np, pandas as pd
from pufferfish.features.build import build_features
from pufferfish.models import evaluate as ev
from pufferfish.models.estimators import SeasonalRate, candidate


def test_chrono_splits_never_train_on_the_future():
    for tr, te in ev.chrono_splits(250):
        assert tr.max() < te.min()


def test_seasonal_baseline_is_higher_in_peak_season(hauls):
    df = build_features(hauls)
    m = SeasonalRate().fit(df[["doy_sin", "doy_cos"]], df.puffer)
    winter = build_features(hauls.iloc[:1].assign(date=pd.to_datetime("2024-01-10")))
    summer = build_features(hauls.iloc[:1].assign(date=pd.to_datetime("2024-06-10")))
    pw, ps = m.predict_proba(winter[["doy_sin", "doy_cos"]])[0, 1], m.predict_proba(summer[["doy_sin", "doy_cos"]])[0, 1]
    assert 0 < ps < pw < 1


def test_real_signal_beats_prevalence_and_permutation_test_agrees(hauls):
    df = build_features(hauls)
    ev.EV.update(permutations=40, bootstrap=100, cv_repeats=2)
    r = ev.compare_candidates(df, ["prevalence_baseline", "logistic_season2"], with_cv=False, with_perm=True,
                              perm_for=("logistic_season2",))
    a = r.set_index("model")
    assert a.loc["logistic_season2", "chrono_average_precision"] > a.loc["prevalence_baseline", "chrono_average_precision"]
    assert a.loc["logistic_season2", "perm_p_value"] < 0.1


def test_noise_labels_are_not_called_significant(hauls):
    df = build_features(hauls.assign(puffer=np.random.default_rng(5).integers(0, 2, len(hauls))))
    ev.EV.update(permutations=40, bootstrap=100)
    r = ev.compare_candidates(df, ["logistic_season2"], with_cv=False, with_perm=True, perm_for=("logistic_season2",))
    assert r.perm_p_value[0] > 0.05


def test_sst_models_are_skipped_when_sst_missing(hauls):
    r = ev.compare_candidates(build_features(hauls), ["logistic_season_sst"], with_cv=False)
    assert r.status[0].startswith("skipped")


def test_learning_curve_uses_only_real_rows_and_grows():
    from pufferfish.models.evaluate import learning_curve
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    n = 200
    doy = rng.integers(1, 366, n)
    df = pd.DataFrame({"doy_sin": np.sin(2*np.pi*doy/365.25), "doy_cos": np.cos(2*np.pi*doy/365.25),
                       "doy_sin2": np.sin(4*np.pi*doy/365.25), "doy_cos2": np.cos(4*np.pi*doy/365.25)})
    df["puffer"] = (rng.random(n) < 0.1 + 0.4*(df["doy_cos"] > 0.5)).astype(int)
    lc = learning_curve(df, names=("logistic_season2",), fractions=(0.5, 1.0), repeats=3)
    assert set(lc.columns) >= {"train_hauls", "ap_mean", "ap_sd", "no_skill_ap"}
    assert lc["train_hauls"].max() <= int(n * 0.6)
