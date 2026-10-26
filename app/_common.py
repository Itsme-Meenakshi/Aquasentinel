import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
from pufferfish.twin.state import TwinState

NOT_READY = ("The twin has not been built yet. Run these two commands in the project folder, then reload this page:\n\n"
             "`python notebooks/05_data_cleaning.py`\n\n`python scripts/run_pipeline.py`")


def twin() -> TwinState:
    return TwinState()


def require_twin():
    tw = twin()
    if not tw.initialised:
        st.warning(NOT_READY)
        st.stop()
    return tw


@st.cache_data(show_spinner=False)
def observations(_stamp: float) -> pd.DataFrame:
    return twin().observations()


def obs_df() -> pd.DataFrame:
    tw = require_twin()
    return observations(tw.obs_file.stat().st_mtime)


def show_map(m, height=520):
    """Render a folium map. Works across Streamlit versions (st.iframe is new; components.html is the old way)."""
    html = m._repr_html_()
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


def show_table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:                      # older Streamlit
        st.dataframe(df, use_container_width=True, hide_index=True)
