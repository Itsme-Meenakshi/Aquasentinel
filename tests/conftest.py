import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd, pytest


@pytest.fixture
def hauls():
    """Synthetic hauls with a known seasonal pattern (not real data)."""
    rng = np.random.default_rng(1)
    dates = pd.to_datetime("2023-09-01") + pd.to_timedelta(np.sort(rng.integers(0, 1000, 240)), unit="D")
    doy = dates.day_of_year.to_numpy()
    p = np.where((doy > 305) | (doy < 60), 0.45, 0.04)
    return pd.DataFrame({"date": dates, "season": "x", "lat": 9.85 + rng.normal(0, .08, 240),
                         "lon": 76.17 + rng.normal(0, .04, 240), "depth": rng.uniform(5, 30, 240),
                         "gear": rng.choice(["Ring seine", "Mini ring seine"], 240),
                         "puffer": (rng.random(240) < p).astype(int)})
