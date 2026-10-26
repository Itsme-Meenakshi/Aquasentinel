# System architecture

```
CMFRI Excel  -> notebooks/05_data_cleaning.py -> data/processed/pufferfish_cleaned.csv
                                                        |
NOAA OISST (ERDDAP or manual CSV) -> environmental/sst.py (nearest ocean pixel, pre-haul window, match quality)
                                                        |
                                   features/build.py -> data/processed/hauls_with_features.csv
                                                        |
                         models/evaluate.py (chronological, bootstrap, permutation) -> reports/*.csv, *.png
                                                        |
                                   twin/state.py  <-> artifacts/twin/ (observations, model, pending, history)
                                                        |
                                   app/ (Streamlit): explorer, results, scenario, report a haul, history
```
Review loop: submitted haul -> validation -> pending -> human accepts/rejects -> retrain -> new version in history.
