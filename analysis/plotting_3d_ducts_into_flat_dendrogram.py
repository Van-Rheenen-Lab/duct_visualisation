import matplotlib.pyplot as plt
import networkx as nx
from utils.loading_saving import load_duct_systems, create_duct_graph, save_annotations
from utils.plotting_trees import plot_hierarchical_graph
from utils.fixing_annotations import connect_component_to_main, simplify_duct_system
from utils.plotting_3d import plot_3d_system

# Load duct systems
duct_systems = load_duct_systems(
    r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations hendrik\A18.json")

duct_graphs = []
for idx, system_data in duct_systems.items():
    G = create_duct_graph(system_data)
    if len(G.nodes) > 0:
        duct_graphs.append((G, system_data))

# Take the first graph for demonstration
G, system_data = duct_graphs[0]


# Try to unify disconnected components
while True:
    connected_components = list(nx.connected_components(G.to_undirected()))
    if len(connected_components) <= 1:
        break

    merged_any = False
    sorted_components = sorted(connected_components, key=len, reverse=True)
    main_component = sorted_components[0]

    for comp in sorted_components[1:]:
        if connect_component_to_main(G, main_component, comp, system_data, coord_tol=10):
            merged_any = True

    if not merged_any:
        break

from utils.plotting_trees import create_annotation_color_map
annotation_to_color = create_annotation_color_map(system_data)

## Request: set specific colors for specific annotations, just overwrite the annotation_to_color dictionary
# annotation_to_color = {"Abnormal": "#FF0000", "Endpoint": "#0080FE"}

# After merging
plot_3d_system(G, system_data, annotation_to_color)
print("Plotting hierarchical after merging:")
plot_hierarchical_graph(G, system_data=system_data, annotation_to_color=annotation_to_color, use_hierarchy_pos=True, orthogonal_edges=True, vert_gap=1, vert_length=1)

# save_annotations(duct_systems, "fixed_annotations.json")
plt.show()