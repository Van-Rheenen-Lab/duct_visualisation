import matplotlib.pyplot as plt
import networkx as nx
from utils.loading_saving import load_duct_systems, create_duct_graph
from utils.plotting_trees import plot_hierarchical_graph, create_annotation_color_map
from utils.fixing_annotations import connect_component_to_main
from utils.plotting_3d import plot_3d_system

duct_systems = load_duct_systems(
    r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\Coseq.json"
)

duct_graphs = []
for idx, system_data in duct_systems.items():
    # Build an undirected graph with full node attributes.
    G = create_duct_graph(system_data)
    if len(G.nodes) > 0:
        duct_graphs.append(G)

# Take the first graph for demonstration.
G = duct_graphs[0]

# (Optional) Unify disconnected components.
while True:
    connected_components = list(nx.connected_components(G.to_undirected()))
    if len(connected_components) <= 1:
        break

    merged_any = False
    sorted_components = sorted(connected_components, key=len, reverse=True)
    main_component = sorted_components[0]

    for comp in sorted_components[1:]:
        # Ensure connect_component_to_main is updated to work directly on G.
        if connect_component_to_main(G, main_component, comp, coord_tol=10):
            merged_any = True

    if not merged_any:
        break

# Generate a color map based on annotations found in the graph.
annotation_to_color = create_annotation_color_map(G, colormap_name="tab10")
# Optionally override with specific colors.
annotation_to_color = {"6": "#00FFFF", "8": "#0080FE", "11": "#00FF00", "15": "#F8F000"}

# Plot the 3D duct system using the updated function.
plot_3d_system(G, annotation_to_color)

print("Plotting hierarchical view after merging:")
fig, ax = plot_hierarchical_graph(
    G,
    annotation_to_color=annotation_to_color,
    use_hierarchy_pos=True,
    orthogonal_edges=True,
    vert_gap=1,
    vert_length=1
)
plt.show()
