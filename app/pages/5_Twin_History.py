import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from _common import require_twin, show_table

st.set_page_config(page_title="Twin history", layout="wide")
st.title("Twin history")
tw = require_twin()
st.caption("Every build, accepted observation and retrain is recorded here, with a fingerprint of the data used.")
show_table(tw.history())
st.subheader("Review log")
st.json(tw.pending())
