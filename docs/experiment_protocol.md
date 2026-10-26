# Experiment protocol

- **Unit:** one haul. **Target:** pufferfish present (1) or confirmed absent (0).
- **Allowed predictors:** only variables known before the haul (see `config/features.yaml`).
- **Splits:** chronological expanding window, first 40% of hauls (by date) as minimum training period, 5 test blocks, pooled.
- **Primary metric:** average precision on later hauls, read against the no-skill level. Secondary: ROC-AUC, Brier skill score.
- **Baselines every model must beat:** prevalence; smoothed seasonal rate.
- **Uncertainty:** 1000 bootstrap resamples of pooled predictions (95% interval). **Significance:** 300 label permutations.
- **Model selection:** highest out-of-time average precision among non-baseline models. Report whether its interval still
  overlaps the seasonal baseline. No test data is used for tuning (hyper-parameters are fixed in code).
- **Claims allowed:** statements supported by the intervals above. Not allowed: coast-wide forecasts, loss estimates,
  statements about causes (temperature or otherwise) without evidence.
- **Re-run when:** new reviewed hauls are accepted and the twin is retrained.
