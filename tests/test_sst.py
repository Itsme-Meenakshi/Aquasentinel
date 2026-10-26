import numpy as np, pandas as pd
from pufferfish.environmental import sst

def grid(spike_day=None):
    dates = pd.date_range("2024-01-01", "2024-03-31")
    g = pd.MultiIndex.from_product([dates, [9.875, 10.125], [76.125, 76.375]], names=["date", "lat", "lon"]).to_frame(index=False)
    g["sst"] = 28.0
    g.loc[(g.lon == 76.125), "sst"] = np.nan            # land-masked pixel next to the coast
    if spike_day is not None:
        g.loc[g.date == spike_day, "sst"] = 99.0
    return g

H = pd.DataFrame({"date": [pd.Timestamp("2024-02-20")], "lat": [9.88], "lon": [76.14]})

def test_falls_back_to_nearest_ocean_pixel():
    f = sst.match_hauls_to_sst(H, grid())
    assert f.sst_status[0] == "matched" and f.sst_pixel_lon[0] == 76.375 and abs(f.sst_prev7[0] - 28.0) < 1e-9
    assert 0 < f.sst_match_km[0] < 60

def test_never_uses_the_haul_day_or_later():
    f = sst.match_hauls_to_sst(H, grid(spike_day=pd.Timestamp("2024-02-20")))
    assert abs(f.sst_prev7[0] - 28.0) < 1e-9                  # the 99.0 on the haul day must be ignored

def test_far_hauls_are_flagged_not_guessed():
    far = H.assign(lat=11.9, lon=75.4)
    f = sst.match_hauls_to_sst(far, grid())
    assert np.isnan(f.sst_prev7[0]) and f.sst_status[0] in ("no_pixel_within_limit", "insufficient_days")

def test_parse_erddap_skips_units_row():
    txt = "time,zlev,latitude,longitude,sst\nUTC,m,degrees_north,degrees_east,degree_C\n2023-09-01T12:00:00Z,0.0,9.875,76.375,28.9\n"
    d = sst.parse_erddap_csv(txt)
    assert len(d) == 1 and d.sst[0] == 28.9 and d.date[0] == pd.Timestamp("2023-09-01")

def test_fetch_uses_getter_and_caches(tmp_path):
    calls = []
    def getter(url):
        calls.append(url)
        class R:
            status_code = 200
            text = "time,zlev,latitude,longitude,sst\nUTC,m,degrees_north,degrees_east,degree_C\n2024-01-01T12:00:00Z,0.0,9.875,76.375,28.9\n2024-01-02T12:00:00Z,0.0,9.875,76.375,29.0\n"
        return R()
    f = tmp_path / "c.csv"
    sst.fetch_oisst("2024-01-01", "2024-01-02", cache_file=f, getter=getter)
    sst.fetch_oisst("2024-01-01", "2024-01-02", cache_file=f, getter=getter)
    assert len(calls) == 1 and f.exists()


def test_retries_then_falls_back_to_second_server(tmp_path, monkeypatch):
    monkeypatch.setattr(sst.time, "sleep", lambda s: None)
    seen = []
    def getter(url):
        seen.append(url)
        if "coastwatch" in url:
            raise TimeoutError("slow server")
        class R:
            status_code = 200
            text = "time,zlev,latitude,longitude,sst\nUTC,m,degrees_north,degrees_east,degree_C\n2024-01-01T12:00:00Z,0.0,9.875,76.375,28.9\n"
        return R()
    g = sst.fetch_oisst("2024-01-01", "2024-01-01", cache_file=tmp_path / "c.csv", getter=getter)
    assert len(g) == 1 and sum("coastwatch" in u for u in seen) == 3 and "upwell" in seen[-1]


def test_manual_url_is_browser_safe_and_manual_csv_loads(tmp_path):
    u = sst.manual_download_url("2023-08-01", "2023-08-31")
    assert "[" not in u and "%5B" in u and "ncdcOisst21Agg_LonPM180" in u
    f = tmp_path / "m.csv"
    f.write_text("time,zlev,latitude,longitude,sst\nUTC,m,degrees_north,degrees_east,degree_C\n2023-08-01T12:00:00Z,0.0,9.875,76.375,28.9\n")
    g = sst.load_manual_csv(f)
    assert len(g) == 1 and g.sst[0] == 28.9
