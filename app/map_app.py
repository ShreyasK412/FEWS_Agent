import os, json
import streamlit as st
import folium, requests
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Ethiopia Food Security (Prototype)")

BOUNDARY_PATH = "data/geo/ethiopia_admin0.geojson"
CENTER = [9.145, 40.489]
IPC_COLORS = {
    "Phase 1: Minimal": "#00A651",
    "Phase 2: Stressed": "#FFD100",
    "Phase 3: Crisis": "#F36C21",
    "Phase 4: Emergency": "#E31A1C",
    "Phase 5: Famine": "#6A1E14"
}

m = folium.Map(location=CENTER, zoom_start=6, tiles="cartodbpositron", control_scale=True)
if os.path.exists(BOUNDARY_PATH):
    with open(BOUNDARY_PATH, "r") as f:
        gj = json.load(f)
    folium.GeoJson(gj, name="Ethiopia", style_function=lambda feat: {
        "color": "#2E3A59", "weight": 2, "fillOpacity": 0.0
    }).add_to(m)
else:
    folium.Rectangle(bounds=[[3.4,33.0],[14.9,48.0]], color="#2E3A59", weight=2, fill=False,
                     tooltip="Ethiopia (approx extent)").add_to(m)

title_html = """
<div style="position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
     z-index: 9999; background: rgba(255,255,255,0.96);
     padding: 8px 14px; border-radius: 10px;
     border: 1px solid #e3e3e3;
     font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
     box-shadow: 0 2px 10px rgba(0,0,0,0.15);">
  <div style="font-weight: 700; font-size: 16px; letter-spacing: .2px; color:#2E3A59;">
    Ethiopia — Acute Food Insecurity Area Classification (Prototype)
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

from jinja2 import Template
legend_html = """<div id='ipc-legend' style="position: fixed; 
     bottom: 20px; left: 20px; z-index: 9999; 
     background: rgba(255,255,255,0.96); padding: 10px 12px; 
     border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);
     font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
     border: 1px solid #e3e3e3;">
  <div style="font-weight: 600; margin-bottom: 6px; font-size: 13px;">
    Acute Food Insecurity Area Classification (IPC)
  </div>
  {% for label, color in items %}
    <div style="display: flex; align-items: center; margin: 3px 0; font-size: 12px;">
      <span style="display:inline-block; width:16px; height:12px; background: {{color}}; border: 1px solid #999; margin-right:8px;"></span>
      <span>{{label}}</span>
    </div>
  {% endfor %}
</div>
"""
legend = Template(legend_html).render(items=IPC_COLORS.items())
m.get_root().html.add_child(folium.Element(legend))

st_data = st_folium(m, width=1200, height=700)

st.sidebar.header("Ask the agent")
q = st.sidebar.text_area("Question", "What are drivers of IPC3 in Amhara?")
if st.sidebar.button("Search"):
    try:
        r = requests.get("http://localhost:8000/search-text", params={"q": q})
        r.raise_for_status()
        results = r.json()
        for i, p in enumerate(results, 1):
            st.markdown(f"**{i}.** {p.get('text','')[:400]} ...  
"
                        f"_source: {p.get('source')} • date: {p.get('round_date')} • file: {p.get('file')}_")
    except Exception as e:
        st.error(f"Error: {e}")
