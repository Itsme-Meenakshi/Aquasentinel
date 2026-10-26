"""Interactive folium maps (rendered in Streamlit with components.html)."""
import folium
import pandas as pd


def haul_map(df: pd.DataFrame, colour_by="puffer", tiles="OpenStreetMap") -> folium.Map:
    """Recorded hauls: red = pufferfish present, blue = confirmed absent."""
    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=10, tiles=tiles)
    for _, r in df.iterrows():
        present = int(r["puffer"]) == 1
        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=5 if present else 3, weight=1,
            color="#C00000" if present else "#1f4e79", fill=True, fill_opacity=0.8 if present else 0.45,
            tooltip=f"{pd.Timestamp(r['date']).date()} | depth {r['depth']:.1f} m | {r['gear']} | "
                    f"{'pufferfish present' if present else 'no pufferfish'}").add_to(m)
    return m


def scenario_map(df: pd.DataFrame, lat, lon, coverage_km, label="Scenario position") -> folium.Map:
    """Observed hauls plus the scenario position and the coverage radius inside which the twin will answer."""
    m = haul_map(df)
    folium.Circle([lat, lon], radius=coverage_km * 1000, color="#555", weight=1, fill=False,
                  tooltip=f"{coverage_km} km coverage radius").add_to(m)
    folium.Marker([lat, lon], tooltip=label, icon=folium.Icon(color="green")).add_to(m)
    return m
