# Data dictionary

Source: `2023-26 Ring seine haul Puffer Meta Data.xlsx` (CMFRI, supplied by Mohammed Jabir). 253 rows, one per haul.

| Column | Meaning | Use |
|---|---|---|
| Date | haul date (dd.mm.yyyy) | predictor (day of year); authoritative over the Month column |
| Month, Season | labels given by the provider | Month is re-derived from Date (1 row disagreed); Season not used directly |
| Latitude, Longitude | haul position, decimal degrees | predictor, SST matching, maps |
| Depth (m) | depth of the operation | predictor (available before fishing) |
| Gear | Ring seine / Mini ring seine / Large scale ring seine | predictor (spelling variants merged) |
| PUFFER Presence (1/0) | **target**. 0 = confirmed absence (not missing data), 1 = present | label |
| Duration (Hrs), Total catch (Kg), Major Catch (Species), CPUE | known only after the haul | **excluded** (leakage) |
| Puffer Species | species when present (L. spadiceus 20, L. inermis 8, L. guentheri 7, L. lunaris 3) | **excluded** (it is the target in disguise); descriptive use only |

Added by the pipeline (`data/processed/hauls_with_features.csv`): doy_sin/doy_cos (+ second harmonic), gear_mini, sst_prev7
(mean OISST over the 7 days before the haul), sst_trend14, and matching-quality columns (sst_match_km, sst_status).
