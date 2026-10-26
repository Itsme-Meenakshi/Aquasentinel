"""Run the whole AquaSentinel pipeline:  cleaned hauls -> SST features -> model comparison -> twin state.

  python scripts/run_pipeline.py                       # download NOAA OISST SST and use it
  python scripts/run_pipeline.py --sst-csv file.csv    # use a manually downloaded SST grid instead
  python scripts/run_pipeline.py --skip-sst            # season / location / depth models only
  python scripts/run_pipeline.py --quick               # fewer repeats and permutations (for testing)

Prerequisite: python notebooks/05_data_cleaning.py   (creates data/processed/pufferfish_cleaned.csv)
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from pufferfish.config import CFG, PROJECT_ROOT
from pufferfish.data.loader import load_hauls
from pufferfish.features.build import build_features
from pufferfish.environmental import sst as sstmod
from pufferfish.models import evaluate as ev, report
from pufferfish.models.estimators import CANDIDATES
from pufferfish.models.train import fit_final
from pufferfish.twin.state import TwinState

ap = argparse.ArgumentParser()
ap.add_argument("--sst-csv"); ap.add_argument("--skip-sst", action="store_true")
ap.add_argument("--refresh-sst", action="store_true"); ap.add_argument("--quick", action="store_true")
ap.add_argument("--require-sst", action="store_true", help="fail instead of continuing without SST")
a = ap.parse_args()
if a.quick:
    ev.EV.update(cv_repeats=2, permutations=40, bootstrap=200)

reports, artifacts = PROJECT_ROOT / "reports", PROJECT_ROOT / "artifacts"
(artifacts / "metrics").mkdir(parents=True, exist_ok=True)

print("1/5  Loading hauls ...")
hauls = load_hauls()
print(f"     {len(hauls)} hauls, {int(hauls.puffer.sum())} with pufferfish ({hauls.puffer.mean():.1%}), "
      f"{hauls.date.min().date()} to {hauls.date.max().date()}")

print("2/5  Sea-surface temperature ...")
sst_feats = None
if not a.skip_sst:
    try:
        if a.sst_csv:
            grid = sstmod.load_manual_csv(a.sst_csv)
        else:
            grid = sstmod.fetch_oisst(hauls.date.min() - pd.Timedelta(days=20), hauls.date.max(), refresh=a.refresh_sst)
        sst_feats = sstmod.match_hauls_to_sst(hauls, grid)
        rep = sstmod.matching_report(sst_feats)
        (reports / "sst_matching_report.json").write_text(json.dumps(rep, indent=2, default=float))
        print("    ", rep)
    except Exception as e:  # network blocked, server changed, bad file ...
        if a.require_sst:
            raise
        print(f"     SST unavailable ({type(e).__name__}: {str(e)[:200]}).\n"
              "     Continuing WITHOUT temperature.\n"
              "     Fix: open this link in a browser (try another network, e.g. a phone hotspot), save the CSV, then run\n"
              "          python scripts/run_pipeline.py --sst-csv path\\to\\file.csv\n     "
              + sstmod.manual_download_url(hauls.date.min() - pd.Timedelta(days=20), hauls.date.max()))
df = build_features(hauls, sst_feats)
df.to_csv(PROJECT_ROOT / "data" / "processed" / "hauls_with_features.csv", index=False)

print("3/5  Comparing models on later hauls (this takes a few minutes) ...")
names = list(CANDIDATES)
perm_for = ("seasonal_baseline", "logistic_season2", "logistic_season_depth", "logistic_season_sst",
            "logistic_season_depth_sst")
res = ev.compare_candidates(df, names, with_cv=True, with_perm=True, perm_for=perm_for)
res.to_csv(reports / "model_comparison.csv", index=False)
show = res[res.status == "ok"][["model", "n_predictors", "chrono_average_precision", "chrono_ap_ci_low", "chrono_ap_ci_high",
                                "chrono_roc_auc", "chrono_brier_skill", "cv_ap_mean", "perm_p_value"]]
print(show.round(3).to_string(index=False))

print("3b/5 Learning curve (real hauls only) ...")
lc = ev.learning_curve(df, repeats=5 if a.quick else 30)
lc.to_csv(reports / "learning_curve.csv", index=False)
report.plot_learning_curve(lc, reports / "learning_curve.png")
print(lc.round(3).to_string(index=False))

print("4/5  Figures ...")
report.plot_seasonal_signal(df, reports / "seasonal_signal.png")
report.plot_model_comparison(res, reports / "model_comparison.png")

# select by ranking skill (AP) among models whose probabilities beat a constant forecast (Brier skill > 0);
# the twin reports probabilities, so a model with over-confident probabilities is not eligible.
ok = res[(res.status == "ok") & (~res.model.str.contains("baseline")) & (res.chrono_brier_skill > 0)].sort_values("chrono_average_precision", ascending=False)
best = ok.iloc[0]
seas = res[res.model == "seasonal_baseline"].iloc[0]
summary = {"selected_model": best.model, "selected_chrono_ap": round(float(best.chrono_average_precision), 3),
           "seasonal_baseline_ap": round(float(seas.chrono_average_precision), 3),
           "no_skill_ap": round(float(best.chrono_prevalence), 3),
           "beats_seasonal_baseline": bool(best.chrono_average_precision > seas.chrono_average_precision),
           "ci_overlaps_seasonal_baseline": bool(best.chrono_ap_ci_low < seas.chrono_average_precision)}
(artifacts / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
print("     ", summary)
fit_final(df, best.model)

print("5/5  Initialising the twin state ...")
tw = TwinState()
rec = tw.initialise(df)
print(f"     twin version {rec['version']} | model {rec['model']} | out-of-time AP {rec['out_of_time_ap']} "
      f"(seasonal baseline {rec['seasonal_baseline_ap']}, base rate {rec['base_rate']})")
print("\nDone. Results: reports/model_comparison.csv, reports/*.png.  Dashboard: streamlit run app/main.py")
