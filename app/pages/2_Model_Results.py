import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, streamlit as st
from _common import ROOT, twin, require_twin, show_table

st.set_page_config(page_title="Model Results", layout="wide")
st.title("Model results (tested on LATER hauls)")
require_twin()
f = ROOT / "reports" / "model_comparison.csv"
if not f.exists():
    st.warning("Run `python scripts/run_pipeline.py` to create the comparison."); st.stop()
res = pd.read_csv(f)
ok = res[res["status"] == "ok"].copy()
base = ok.iloc[0]["chrono_prevalence"]
st.markdown(
    f"Each model was trained on earlier hauls and scored on the next block of hauls, repeated 5 times. "
    f"**Average precision** measures how well the model ranks pufferfish hauls above the rest. "
    f"A model with no skill scores about **{base:.2f}** (the share of positive hauls).")
st.image(str(ROOT / "reports" / "model_comparison.png"))
cols = {"model": "Model", "n_predictors": "Predictors", "chrono_average_precision": "Avg precision (later hauls)",
        "chrono_ap_ci_low": "95% low", "chrono_ap_ci_high": "95% high", "chrono_roc_auc": "ROC-AUC",
        "chrono_brier_skill": "Brier skill", "cv_ap_mean": "Avg precision (random folds, optimistic)",
        "perm_p_value": "Permutation p-value"}
show_table(ok[[c for c in cols if c in ok]].rename(columns=cols).round(3))
st.info(
    "How to read this: the permutation p-value asks how often shuffled labels would score this well. Small values "
    "(below about 0.05) mean the seasonal pattern is not luck. The wide 95% ranges show how little data there is: "
    "treat every number as preliminary. Models that add location or depth do not beat season alone.")
st.subheader("Seasonal pattern in the data")
st.image(str(ROOT / "reports" / "seasonal_signal.png"))
