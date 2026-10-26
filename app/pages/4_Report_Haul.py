import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datetime as dt
import pandas as pd, streamlit as st
from _common import require_twin, obs_df
from pufferfish.twin.state import GEARS

st.set_page_config(page_title="Report a haul", layout="wide")
st.title("Report a haul and review submissions")
tw = require_twin()
df = obs_df()
st.caption("New hauls are held for review. They change the twin only after a reviewer accepts them and the model is retrained.")

with st.form("haul"):
    a, b, c = st.columns(3)
    date = a.date_input("Date", dt.date.today())
    lat = b.number_input("Latitude", 8.0, 12.0, float(round(df["lat"].median(), 4)), 0.0001, format="%.4f")
    lon = c.number_input("Longitude", 75.0, 77.5, float(round(df["lon"].median(), 4)), 0.0001, format="%.4f")
    d, e, f = st.columns(3)
    depth = d.number_input("Depth (m)", 1.0, 100.0, 12.0, 0.5)
    gear = e.selectbox("Gear", GEARS)
    puffer = f.selectbox("Pufferfish in the haul?", ["No (confirmed absent)", "Yes (present)"])
    who = st.text_input("Your name")
    if st.form_submit_button("Submit for review"):
        ok, errs, sid = tw.submit({"date": str(date), "lat": lat, "lon": lon, "depth": depth, "gear": gear,
                                   "puffer": int(puffer.startswith("Yes"))}, who or "anonymous")
        st.success(f"Submitted (id {sid}). Waiting for review.") if ok else st.error("; ".join(errs))

st.subheader("Pending submissions")
pend = [p for p in tw.pending() if p["status"] == "pending"]
if not pend:
    st.write("Nothing waiting.")
for p in pend:
    with st.container(border=True):
        st.write(f"**{p['id']}** from {p['submitted_by']}: {p['record']}")
        r1, r2, r3 = st.columns([2, 1, 1])
        rev = r1.text_input("Reviewer name", key=f"rv{p['id']}")
        if r2.button("Accept", key=f"a{p['id']}") and rev:
            tw.review(p["id"], True, rev); st.rerun()
        if r3.button("Reject", key=f"r{p['id']}") and rev:
            tw.review(p["id"], False, rev); st.rerun()
st.subheader("Update the twin")
st.write("After accepting hauls, retrain to create a new version. Out-of-time scores are recomputed and logged.")
if st.button("Retrain now"):
    rec = tw.retrain()
    st.success(f"Version {rec['version']} created with {rec['n_hauls']} hauls. "
               f"Out-of-time average precision {rec['out_of_time_ap']}.")
