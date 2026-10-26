# AquaSentinel: pufferfish occurrence monitoring for the central Kerala coast

BTech major project, Group 15, MITS Kochi (Dept. of AI & Data Science). Guide: Mrs. Sheena Y.
Team: Allan Nizar, Meenakshi Santhosh M S, Naseen E A, P Shreyaa.

**Question:** given when and where a ring-seine haul is made, how likely is it to catch pufferfish?
**Data:** 253 CMFRI ring-seine hauls near Kochi, Sep 2023 to Jun 2026; 38 with pufferfish, 215 confirmed absences.

## What we found so far (preliminary; 38 positive hauls)
- Pufferfish occurrence is **strongly seasonal**: about 32% of November-February hauls had pufferfish, against about 1% of hauls in
  April-July and September-October (`reports/seasonal_signal.png`).
- Tested on **later** hauls, a logistic model on time of year reaches average precision **0.40** (95% interval 0.24-0.59),
  against **0.16** for a model with no skill, and 0.33 for a plain seasonal-rate baseline. A label-permutation test gives
  p = 0.003: the seasonal pattern is not luck. The intervals are wide, so treat the numbers as preliminary.
- **Depth, position and gear add nothing** in this data (Random Forest with them scores below the seasonal baseline).
- Sea-surface temperature is implemented but **its effect is not yet measured** (see Run it).

See `reports/model_comparison.csv` and `docs/experiment_protocol.md` for how every number is produced.

## What the "digital twin" is here
A saved, versioned state (observations + model); reviewed updates (new hauls wait for a human to accept them); a history log
with data fingerprints and out-of-time scores; what-if queries with ranges; and refusal ("insufficient data") where the data
cannot support an answer. It is not a physical simulation of the sea.

## Run it (Windows PowerShell; Python 3.10+)
```
pip install -r requirements.txt
python notebooks/05_data_cleaning.py          # needs the Excel file in data/raw/
python scripts/run_pipeline.py                # downloads NOAA OISST SST, compares models, builds the twin
streamlit run app/main.py
pytest
```
Options: `--skip-sst` (no temperature), `--sst-csv file.csv` (use a manually downloaded SST grid), `--quick` (fast test run).
If the SST download fails the script says so and continues without temperature. Check `config/config.yaml` (server and
dataset name) or download the grid manually and use `--sst-csv`.

## Limits (say these out loud)
- All hauls lie near 9-11 N, 75.8-76.4 E, depth 4-44 m. No claim is made for the rest of Kerala.
- The data has no net-damage or economic-loss fields, so none are predicted.
- Hauls are one fishing ground over three years; later hauls differ from earlier ones, so out-of-time scores are modest.
- Seasonality may partly reflect when and how the fleet fishes, not only the fish.

## Layout
`notebooks/` audit, EDA, cleaning | `src/pufferfish/` data, features, environmental, models, geo, twin | `scripts/run_pipeline.py` |
`app/` Streamlit | `config/` settings | `docs/` methodology, data dictionary, protocol | `reports/` figures and tables | `tests/`
