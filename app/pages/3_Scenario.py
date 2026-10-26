import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datetime as dt
import streamlit as st
from _common import require_twin, obs_df, show_map
from pufferfish.config import CFG
from pufferfish.geo.maps import scenario_map
from pufferfish.twin.state import GEARS

st.set_page_config(page_title="Scenario", layout="wide")
st.title("What-if scenario")
tw = require_twin()
df = obs_df()
st.caption("Ask for the chance of pufferfish in a haul. The twin answers only where the data supports it.")
c1, c2, c3 = st.columns(3)
date = c1.date_input("Planned haul date", dt.date.today())
lat = c2.number_input("Latitude", 8.0, 12.0, float(round(df["lat"].median(), 3)), 0.01, format="%.3f")
lon = c3.number_input("Longitude", 75.0, 77.5, float(round(df["lon"].median(), 3)), 0.01, format="%.3f")
c4, c5, c6 = st.columns(3)
depth = c4.number_input("Depth (m)", 1.0, 100.0, float(round(df["depth"].median(), 1)), 0.5)
gear = c5.selectbox("Gear", GEARS)
use_sst = c6.checkbox("I know the sea-surface temperature")
sst = c6.number_input("SST over the last 7 days (C)", 20.0, 34.0, 29.0, 0.1) if use_sst else None

r = tw.predict(date, lat, lon, depth, gear, sst)
if r["status"] == "insufficient_data":
    st.error(f"Insufficient data: {r['reason']}. No probability is shown.")
else:
    st.metric("Chance of pufferfish in the haul", f"{r['p']:.0%}", f"90% range {r['p_low']:.0%} to {r['p_high']:.0%}", delta_color="off")
    st.write(f"For comparison, {r['base_rate']:.0%} of all recorded hauls had pufferfish.")
    if use_sst and "sst_prev7" not in r["features_used"]:
        st.caption(f"The current model ({r['model']}) does not use temperature, so the SST value was ignored.")
st.caption(f"Model: {r['model']} | features: {', '.join(r['features_used'])} | nearest recorded haul: {r['nearest_haul_km']} km | "
           f"hauls within 30 days of this time of year: {r['hauls_same_season']}")
show_map(scenario_map(df, lat, lon, CFG["twin"]["coverage_km"]), 480)
