# Methodology

1. **Audit and clean** (`notebooks/01-07`, `05_data_cleaning.py`): labels, dates, ranges, duplicates, month/date check.
2. **Features known before the haul only** (`src/pufferfish/features`): cyclical day of year, depth, position, gear;
   optional SST from the 7 days before the haul (`src/pufferfish/environmental/sst.py`).
3. **Models** (`src/pufferfish/models`): prevalence baseline; smoothed seasonal baseline; logistic regression with 2-4
   predictors; depth-limited Random Forest. With 38 positives, about 10 positives per predictor allows 3-4 predictors.
4. **Evaluation for small data** (`models/evaluate.py`): trained on earlier hauls, tested on the next block (expanding window,
   5 blocks, pooled). Metrics: average precision (compared with the no-skill level = share of positives), ROC-AUC, Brier skill.
   Uncertainty: bootstrap 95% intervals. Significance: label-permutation test. Random-fold cross-validation is shown only as
   an optimistic stress test.
5. **Twin** (`src/pufferfish/twin`): saved state, reviewed updates, version history, scenario queries with ranges,
   and "insufficient data" outside the observed ground.
6. **Interface** (`app/`): Streamlit, 6 pages. Run with `streamlit run app/main.py`.

Reproduce: `python notebooks/05_data_cleaning.py`, then `python scripts/run_pipeline.py`, then `pytest`.
