import pandas as pd
from pufferfish.config import LEAKAGE_COLUMNS
from pufferfish.data.loader import load_hauls, normalise_gear
from pufferfish.features.build import build_features


def test_gear_spelling_variants_are_merged():
    g = normalise_gear(pd.Series(["Mini Ring Seine", "Mini ring seine", "Ring seine", "Large Scale Ring seine"]))
    assert g.tolist() == ["Mini ring seine", "Mini ring seine", "Ring seine", "Large ring seine"]


def test_loader_drops_post_haul_columns(tmp_path):
    cols = ["Date", "Month", "Season", "Latitude", "Longitude", "Depth (m)", "Gear", "Duration (Hrs)", "Total catch (Kg)",
            "Major Catch (Species)", "PUFFER Presence (1/0)", "Puffer Species",
            "CPUE(catch per unit effoert) Total Catch (kg/hr)"]
    row = ["2024-01-05", "January", "Post-monsoon", 9.8, 76.2, 10.0, "Ring seine", 1.5, 1000, "Sardine", 1, "L. inermis", 666.0]
    f = tmp_path / "c.csv"; pd.DataFrame([row], columns=cols).to_csv(f, index=False)
    df = load_hauls(f)
    assert not set(LEAKAGE_COLUMNS) & set(df.columns)
    assert list(df.columns) == ["date", "season", "lat", "lon", "depth", "gear", "puffer"]


def test_december_and_january_are_neighbours(hauls):
    X = build_features(hauls.iloc[:2].assign(date=pd.to_datetime(["2023-12-31", "2024-01-01"])))
    assert abs(X.doy_sin[0] - X.doy_sin[1]) < 0.05 and abs(X.doy_cos[0] - X.doy_cos[1]) < 0.05
