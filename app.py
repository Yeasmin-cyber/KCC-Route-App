import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import folium_static
import torch
import torch.nn as nn
import random  

# --- 1. Page Configuration ---
st.set_page_config(page_title="KCC Route Optimization", layout="wide")
st.title("🚛 KCC Fecal Sludge Route Optimization")
st.markdown("**AI-Hybrid Model (GNN + Dijkstra) with Real-Time Traffic Simulation**")

# --- 2. GNN Model Definition ---
class RealTimeTrafficGNN(nn.Module):
    def __init__(self):
        super(RealTimeTrafficGNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return 1.0 + self.sigmoid(self.fc2(x))

ai_model = RealTimeTrafficGNN()

# --- 3. Data & Graph Setup (Cached to load only once) ---
@st.cache_resource
def load_graph_and_nodes():
    fstp_lat, fstp_lon = 22.79398, 89.491823
    
    # All 44 Collection Points in KCC
    collection_coords = [
        (22.897467, 89.501752), (22.894609, 89.515169), (22.864415, 89.526720),
        (22.877751, 89.520525), (22.854738, 89.527598), (22.847730, 89.513618),
        (22.845971, 89.528832), (22.839385, 89.522020), (22.838030, 89.536209),
        (22.850093, 89.547798), (22.839687, 89.548149), (22.820892, 89.544093),
        (22.811549, 89.540319), (22.829615, 89.552474), (22.831872, 89.528635),
        (22.832548, 89.543333), (22.812867, 89.549969), (22.816222, 89.566623),
        (22.818628, 89.555155), (22.825843, 89.537212), (22.774034, 89.576737),
        (22.856356, 89.536461), (22.847220, 89.540241), (22.858290, 89.546450),
        (22.864765, 89.538999), (22.797439, 89.552056), (22.804993, 89.553707),
        (22.811295, 89.559441), (22.805815, 89.566529), (22.802722, 89.545178),
        (22.799376, 89.562830), (22.803131, 89.578140), (22.798268, 89.570445),
        (22.793749, 89.579724), (22.781920, 89.577047), (22.789188, 89.570759),
        (22.810818, 89.573463), (22.885038, 89.498048), (22.890268, 89.507143),
        (22.883686, 89.514196), (22.869835, 89.522035), (22.866276, 89.512400),
        (22.858562, 89.517755), (22.903636, 89.511915)
    ]
    
    ox.settings.use_cache = True
    
    with st.spinner("Downloading KCC Road Network... (Takes about 1 minute)"):
        G_raw = ox.graph_from_point((fstp_lat, fstp_lon), dist=12000, network_type='drive')
        G_un = G_raw.to_undirected()
        largest_cc = max(nx.connected_components(G_un), key=len)
        G_main = G_un.subgraph(largest_cc).copy()
        
        fstp_node = ox.distance.nearest_nodes(G_main, fstp_lon, fstp_lat)
        return G_main, fstp_node, collection_coords, fstp_lat, fstp_lon

G_main, fstp_node, collection_coords, fstp_lat, fstp_lon = load_graph_and_nodes()

# --- 4. User Interface (UI) ---
st.sidebar.header("Navigation Control")
st.sidebar.markdown("Select a point below to simulate a Vacutug route.")

point_options = [f"Collection Point {i+1}" for i in range(len(collection_coords))]
selected_point_name = st.sidebar.selectbox("Current Vacutug Location:", point_options)

selected_idx = int(selected_point_name.split(" ")[-1]) - 1
start_lat, start_lon = collection_coords[selected_idx]

# --- 5. Routing Logic ---
if st.sidebar.button("🚗 Find Optimal Route to FSTP"):
    with st.spinner("Analyzing traffic & calculating AI optimal route..."):
        
        start_node = ox.distance.nearest_nodes(G_main, start_lon, start_lat)
        
        with torch.no_grad():
            for u, v, key, data in G_main.edges(keys=True, data=True):
                length = data.get('length', 10.0)
                if isinstance(length, list): length = length[0]
                
                simulated_congestion = random.uniform(1.0, 3.0) 
                
                features = torch.tensor([[float(length), float(simulated_congestion)]], dtype=torch.float32)
                ai_factor = ai_model(features).item()
                data['ai_weight'] = float(length) * ai_factor

        try:
            route = nx.shortest_path(G_main, source=start_node, target=fstp_node, weight='ai_weight')
            
            # Distance Calculation
            total_length_m = sum(
                G_main[u][v][0]["length"] for u, v in zip(route[:-1], route[1:])
            )
            total_length_km = total_length_m / 1000.0
            
            # Fuel & Cost Calculation
            fuel_consumption_rate = 0.20  # Liters per km
            fuel_price_per_liter = 115    # BDT per Liter
            
            fuel_needed_liters = total_length_km * fuel_consumption_rate
            total_fuel_cost_bdt = fuel_needed_liters * fuel_price_per_liter
            
            st.success(f"✅ Route Found for {selected_point_name}! (Avoiding simulated traffic)")
            
            st.info("ℹ️ **Vehicle Information:** Vacutug Capacity: 2000 Liters | Fuel Efficiency: 0.20 L/km | Fuel Price: BDT 115/L")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(
                label="🛣️ Total Distance", 
                value=f"{total_length_km:.2f} km"
            )
            col2.metric(
                label="⛽ Est. Fuel Needed", 
                value=f"{fuel_needed_liters:.2f} L"
            )
            col3.metric(
                label="💰 Est. Fuel Cost", 
                value=f"BDT {total_fuel_cost_bdt:.2f}"
            )
            
            # Map Visualization
            m = folium.Map(location=[(start_lat + fstp_lat)/2, (start_lon + fstp_lon)/2], zoom_start=13)
            
            folium.Marker([fstp_lat, fstp_lon], popup="FSTP (Rajbandh)", icon=folium.Icon(color="red", icon="trash")).add_to(m)
            folium.Marker([start_lat, start_lon], popup=selected_point_name, icon=folium.Icon(color="blue", icon="truck")).add_to(m)
            
            route_coords = [(G_main.nodes[n]['y'], G_main.nodes[n]['x']) for n in route]
            
            folium.PolyLine(
                route_coords, 
                color="#00aa00", 
                weight=5, 
                opacity=0.8,
                tooltip=f"Vacutug (2000L) | Dist: {total_length_km:.2f} km | Fuel: {fuel_needed_liters:.2f} L | Cost: BDT {total_fuel_cost_bdt:.2f}"
            ).add_to(m)
            
            folium_static(m, width=900, height=500)
            
        except nx.NetworkXNoPath:
            st.error("Error: Could not find a connected path.")
else:
    st.info("👈 Please select a Collection Point from the sidebar and click the button to start.")
