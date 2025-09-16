import streamlit as st
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import geopandas as gpd
import json
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from jinja2 import Template

# Set page config
st.set_page_config(
    layout="wide",
    page_title="FEWS NET Ethiopia - Food Insecurity Classification",
    page_icon="🌍"
)

# Custom CSS for FEWS NET styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .control-panel {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🌍 FEWS NET Ethiopia</h1>
    <p>Acute Food Insecurity Area Classification</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_ethiopia_data():
    """Load Ethiopia administrative boundaries"""
    try:
        # Try multiple data sources for Ethiopia
        urls = [
            "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson",
            "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/ethiopia.json",
            "https://raw.githubusercontent.com/python-visualization/folium-example-data/main/world_countries.json"
        ]
        
        eth = None
        for url in urls:
            try:
                world_data = gpd.read_file(url)
                # Look for Ethiopia in the data
                if 'name' in world_data.columns:
                    eth = world_data[world_data['name'] == 'Ethiopia']
                elif 'NAME' in world_data.columns:
                    eth = world_data[world_data['NAME'] == 'Ethiopia']
                elif 'country' in world_data.columns:
                    eth = world_data[world_data['country'] == 'Ethiopia']
                
                if not eth.empty:
                    break
            except:
                continue
        
        # If no data found, create a simple Ethiopia boundary
        if eth is None or eth.empty:
            # Create a simple rectangular boundary for Ethiopia as fallback
            from shapely.geometry import Polygon
            ethiopia_bounds = Polygon([
                [32.997583, 3.404166],  # Southwest
                [47.978333, 3.404166],  # Southeast  
                [47.978333, 18.002083], # Northeast
                [32.997583, 18.002083], # Northwest
                [32.997583, 3.404166]   # Close polygon
            ])
            
            eth = gpd.GeoDataFrame(
                {'name': ['Ethiopia'], 'geometry': [ethiopia_bounds]},
                crs='EPSG:4326'
            )
        
        # Ensure it's in the correct CRS
        eth = eth.to_crs(epsg=4326)
        
        return eth
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
eth_data = load_ethiopia_data()

if eth_data is not None:
    # Control Panel
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Date selector
        selected_date = st.date_input(
            "📅 Analysis Date",
            value=datetime.now().date(),
            min_value=datetime(2020, 1, 1).date(),
            max_value=datetime.now().date()
        )
    
    with col2:
        # Map style selector
        map_style = st.selectbox(
            "🗺️ Map Style",
            ["CartoDB positron", "OpenStreetMap", "CartoDB dark_matter"],
            index=0
        )
    
    with col3:
        # Data source selector
        data_source = st.selectbox(
            "📊 Data Source",
            ["IPC Analysis", "FEWS NET", "WFP", "Combined Sources"],
            index=0
        )
    
    with col4:
        # Population filter
        show_population = st.checkbox("👥 Show Population", value=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create map
    if len(eth_data) > 0:
        # Get center point
        centroid = eth_data.geometry.union_all().centroid
        center_lat, center_lon = centroid.y, centroid.x
    else:
        center_lat, center_lon = 9.145, 40.4897
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles=map_style,
        control_scale=True
    )
    
    # Add fullscreen control
    Fullscreen(position='topleft').add_to(m)
    
    # Add different tile layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    
    # Add Ethiopia boundary with better styling
    style_fn = lambda feature: {
        'fillColor': '#e8f4fd',  # light blue fill
        'color': '#2E3A59',      # dark blue border
        'weight': 3,              # thicker border
        'fillOpacity': 0.6,       # semi-transparent fill
        'dashArray': '5, 5'       # dashed border
    }
    
    # FEWS NET / IPC-style color palette
    IPC_COLORS = {
        "Phase 1: Minimal": "#a4d886",   # green
        "Phase 2: Stressed": "#f6f6a2",  # yellow
        "Phase 3: Crisis": "#f8d28c",    # orange
        "Phase 4: Emergency": "#f29b95", # red
        "Phase 5: Famine": "#7f181d"     # dark red
    }
    
    # Add Ethiopia boundary
    folium.GeoJson(
        data=json.loads(eth_data.to_json()),
        name='Ethiopia',
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Country'])
    ).add_to(m)
    
    # Add sample regional data for demonstration
    # These are approximate coordinates for major Ethiopian regions
    sample_regions = [
        {"name": "Tigray", "lat": 14.1, "lon": 38.7, "phase": 5, "population": 5.4},
        {"name": "Afar", "lat": 11.8, "lon": 41.2, "phase": 4, "population": 1.8},
        {"name": "Amhara", "lat": 11.5, "lon": 37.5, "phase": 3, "population": 22.0},
        {"name": "Oromia", "lat": 8.5, "lon": 39.0, "phase": 4, "population": 35.5},
        {"name": "Somali", "lat": 7.0, "lon": 44.0, "phase": 5, "population": 5.7},
        {"name": "SNNP", "lat": 6.5, "lon": 37.5, "phase": 3, "population": 20.0},
        {"name": "Gambela", "lat": 8.0, "lon": 34.5, "phase": 3, "population": 0.4},
        {"name": "Harari", "lat": 9.3, "lon": 42.1, "phase": 1, "population": 0.2},
        {"name": "Addis Ababa", "lat": 9.0, "lon": 38.7, "phase": 1, "population": 3.4},
        {"name": "Dire Dawa", "lat": 9.6, "lon": 41.9, "phase": 2, "population": 0.5}
    ]
    
    # Add markers for each region with IPC phase colors
    for region in sample_regions:
        phase = region["phase"]
        color = list(IPC_COLORS.values())[phase - 1] if 1 <= phase <= 5 else "#cccccc"
        
        folium.CircleMarker(
            location=[region["lat"], region["lon"]],
            radius=8,
            popup=f"""
            <b>{region['name']}</b><br>
            IPC Phase: {phase}<br>
            Population: {region['population']}M
            """,
            color='white',
            weight=2,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div id='ipc-legend' style="position: fixed; 
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
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display the map
    st.subheader("🗺️ Interactive Map")
    
    # Use st_folium to display the map
    map_data = st_folium(m, width=1200, height=700, returned_objects=["last_object_clicked"])
    
    # Show selected date
    st.info(f"Selected analysis date: {selected_date.strftime('%B %d, %Y')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 12px;">
        <p>🌍 <strong>FEWS NET Ethiopia</strong> - Acute Food Insecurity Area Classification</p>
        <p>Data sources: Natural Earth, FEWS NET, IPC | Last updated: """ + selected_date.strftime("%B %d, %Y") + """</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Unable to load Ethiopia data. Please check your internet connection and try again.")
