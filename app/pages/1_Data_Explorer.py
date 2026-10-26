import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from _common import obs_df, show_map, show_table
from pufferfish.geo.maps import haul_map
from pufferfish.models.report import plot_seasonal_signal
import tempfile

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data explorer")
df = obs_df()
f1, f2, f3 = st.columns(3)
years = f1.multiselect("Year", sorted(df["date"].dt.year.unique()), default=sorted(df["date"].dt.year.unique()))
gears = f2.multiselect("Gear", sorted(df["gear"].unique()), default=sorted(df["gear"].unique()))
show = f3.radio("Show", ["All hauls", "Pufferfish present", "Confirmed absent"], horizontal=True)
d = df[df["date"].dt.year.isin(years) & df["gear"].isin(gears)]
if show == "Pufferfish present":
    d = d[d["puffer"] == 1]
elif show == "Confirmed absent":
    d = d[d["puffer"] == 0]
st.write(f"{len(d)} hauls shown. Red = pufferfish present, blue = confirmed absent.")
if len(d):
    show_map(haul_map(d))
st.subheader("Pufferfish by month (all recorded hauls)")
with tempfile.TemporaryDirectory() as t:
    out = Path(t) / "s.png"; plot_seasonal_signal(df, out); st.image(str(out))
show_table(d.sort_values("date", ascending=False))
