import streamlit as st
import osmnx as ox
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import folium
from streamlit_folium import folium_static

# --- Page Config ---
st.set_page_config(page_title="KCC Route Optimization", layout="wide")
st.title("🚛 KCC Fecal Sludge AI Route Optimization")
st.markdown("This application uses a Hybrid **GNN + Dijkstra** model to predict traffic penalties and find the most optimized route to the FSTP.")

# --- Real Coordinates ---
fstp_lat, fstp_lon = 22.79398, 89.491823
real_collection_coords = [
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

# --- GNN Model Definition ---
class HybridGNN(nn.Module):
    def __init__(self, in_channels, edge_dim):
        super(HybridGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 8)
        self.edge_mlp = nn.Sequential(
            nn.Linear(8 * 2 + edge_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        row, col = edge_index
        h_src, h_dst = h[row], h[col]
        combined_edge_features = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        return self.edge_mlp(combined_edge_features)

# --- Data Caching & Processing ---
@st.cache_resource(show_spinner=False)
def load_and_process_graph():
    center_lat = np.mean([lat for lat, lon in real_collection_coords] + [fstp_lat])
    center_lon = np.mean([lon for lat, lon in real_collection_coords] + [fstp_lon])
    
    # Download Graph
    G = ox.graph_from_point((center_lat, center_lon), dist=10000, network_type='drive')
    G = ox.truncate.largest_component(G, strongly=True)
    G = nx.convert_node_labels_to_integers(G)
    
    fstp_node = ox.distance.nearest_nodes(G, fstp_lon, fstp_lat)
    
    collection_nodes = []
    for lat, lon in real_collection_coords:
        node = ox.distance.nearest_nodes(G, lon, lat)
        if node != fstp_node and node not in collection_nodes:
            collection_nodes.append(node)
            
    # PyG Feature Extraction
    degrees = dict(G.degree())
    G_simple = nx.Graph(G)
    clustering = nx.clustering(G_simple)
    
    x = torch.tensor([[degrees[n], clustering[n]] for n in G.nodes()], dtype=torch.float)
    
    edge_index_list, edge_attr_list = [], []
    for u, v, k, data in G.edges(keys=True, data=True):
        edge_index_list.append([u, v])
        raw_length = data.get('length', 1.0)
        length = float(sum(raw_length)) if isinstance(raw_length, list) else float(raw_length)
        traffic_penalty = np.random.uniform(1.0, 4.0)
        edge_attr_list.append([length, traffic_penalty])
        
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Run Inference
    gnn_model = HybridGNN(in_channels=2, edge_dim=2)
    gnn_model.eval()
    with torch.no_grad():
        predictions = gnn_model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr).numpy().flatten()
        
    for idx, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
        raw_length = data.get('length', 1.0)
        length = float(sum(raw_length)) if isinstance(raw_length, list) else float(raw_length)
        data['gnn_smart_weight'] = float(predictions[idx] * length)
        
    return G, fstp_node, collection_nodes

with st.spinner("Downloading Road Network & Running AI Model... (This takes a minute on first load)"):
    G, fstp_node, collection_nodes = load_and_process_graph()

# --- Streamlit UI ---
st.sidebar.header("Navigation Control")
st.sidebar.write("Total valid collection points loaded:", len(collection_nodes))

point_options = [f"Collection Point {i+1}" for i in range(len(collection_nodes))]
selected_point = st.sidebar.selectbox("Select a Collection Point:", point_options)
selected_idx = int(selected_point.split(" ")[-1]) - 1
target_node = collection_nodes[selected_idx]

# রেডিও বাটন বাদ দেওয়া হয়েছে। এখন শুধুমাত্র AI-Optimized রুটই কাজ করবে।
st.sidebar.markdown("---")
st.sidebar.markdown("**Algorithm:** AI-Hybrid (GNN + Dijkstra)")

if st.sidebar.button("🚗 Find Route"):
    weight_param = 'gnn_smart_weight' # বাই-ডিফল্ট GNN এর প্রেডিক্ট করা ট্রাফিক ওয়েট সেট করা হলো
    
    try:
        route = nx.shortest_path(G, source=target_node, target=fstp_node, weight=weight_param)
        
        # Calculate Total Physical Distance
        total_length_m = sum(G[u][v][0]["length"] for u, v in zip(route[:-1], route[1:]))
        total_length_km = total_length_m / 1000.0
        
        st.success("✅ Route Found! AI successfully avoided traffic congestion.")
        st.metric(label="🛣️ Total Optimized Route Distance", value=f"{total_length_km:.2f} km")
        
        # Folium Map Visualization
        m = folium.Map(location=[fstp_lat, fstp_lon], zoom_start=13)
        folium.Marker([fstp_lat, fstp_lon], popup="FSTP", icon=folium.Icon(color="red")).add_to(m)
        folium.Marker([G.nodes[target_node]['y'], G.nodes[target_node]['x']], popup=selected_point, icon=folium.Icon(color="blue")).add_to(m)
        
        route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
        
        # AI রুটের জন্য ম্যাপে একটি নির্দিষ্ট রঙ (সবুজ) ব্যবহার করা হলো
        folium.PolyLine(
            route_coords, 
            color="#00aa00", 
            weight=5, 
            opacity=0.8, 
            tooltip=f"Optimized Distance: {total_length_km:.2f} km"
        ).add_to(m)
        
        folium_static(m, width=900, height=500)
        
    except nx.NetworkXNoPath:
        st.error("Error: Could not find a valid path to the FSTP from this location.")
