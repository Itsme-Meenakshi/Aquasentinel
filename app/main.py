import streamlit as st
from _common import require_twin, obs_df, ROOT

st.set_page_config(page_title="Pufferfish Digital Twin", layout="wide")
st.title("Pufferfish occurrence: monitoring prototype for the central Kerala coast")
tw = require_twin()
df, meta = obs_df(), tw.meta()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Recorded hauls", len(df))
c2.metric("Hauls with pufferfish", int(df["puffer"].sum()), f"{df['puffer'].mean():.0%} of hauls", delta_color="off")
c3.metric("Period", f"{df['date'].min():%b %Y} to {df['date'].max():%b %Y}")
c4.metric("Twin version", int(meta.get("version", 1)))

st.subheader("What this system does")
st.markdown(
    "- **Question:** will a ring-seine haul at this place and time catch pufferfish? (yes / no, as a probability)\n"
    "- **Data:** CMFRI ring-seine hauls, 2023-26. A 0 means pufferfish were confirmed absent.\n"
    "- **Finding so far:** occurrence is strongly seasonal. See *Model Results* for how much of that a model can use.\n"
    "- **Twin:** the state (observations + model) is versioned. New hauls are reviewed before use, and the model "
    "declines to answer where the data cannot support it."
)
st.subheader("What it does not do")
st.markdown(
    "- It does not forecast the whole Kerala coast. All hauls are near Kochi (about 9-11 N, 75.8-76.4 E).\n"
    "- It does not predict net damage or economic loss: the data has neither.\n"
    "- With only %d positive hauls, results are preliminary." % int(df["puffer"].sum())
)
st.caption("Use the sidebar to move between pages.")
