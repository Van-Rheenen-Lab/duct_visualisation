import os
import matplotlib.pyplot as plt
import networkx as nx
from utils.loading_saving import load_duct_systems, create_duct_graph, select_biggest_duct_system
from utils.plotting_trees import plot_hierarchical_graph, create_annotation_color_map
from utils.fixing_annotations import connect_component_to_main
from utils.plotting_3d import plot_3d_system


"""
The challenge of the following script is that the annotations were not really connected. So sometimes there's 2 points
at the same location, but they are not connected. This script tries to connect them, and plot the duct system.
"""


def merge_disconnected_components(G, coord_tol=10):
    """
    Merge disconnected components of the graph G.
    Continues merging until the graph is fully connected or no merge is possible.
    """
    while True:
        connected_components = list(nx.connected_components(G.to_undirected()))
        if len(connected_components) <= 1:
            break
        merged_any = False
        sorted_components = sorted(connected_components, key=len, reverse=True)
        main_component = sorted_components[0]
        for comp in sorted_components[1:]:
            if connect_component_to_main(G, main_component, comp, coord_tol=coord_tol):
                merged_any = True
        if not merged_any:
            break


def process_duct_system(file_path, annotation_to_color, root_node=None,
                        linewidth=2, legend_offset=-1, fig_size=(7, 5)):
    """
    Load a duct system from file_path, create its graph, merge components,
    and plot a hierarchical view using the provided annotation-to-color mapping.
    Saves the resulting figure at 600 dpi.

    Parameters:
      - file_path: Path to the JSON file.
      - annotation_to_color: Either a fixed dict or a callable for dynamic color mapping.
      - root_node: The root node to use (if provided); otherwise, the system's own value.
      - linewidth: Line width for plotting.
      - legend_offset: Legend offset value.
      - fig_size: Tuple specifying the figure size in inches.
    """
    duct_systems = load_duct_systems(file_path)
    # Use the largest duct system from the file.
    system_data = select_biggest_duct_system(duct_systems)
    G = create_duct_graph(system_data)

    if len(G.nodes) == 0:
        print("Graph is empty for", file_path)
        return

    merge_disconnected_components(G)

    # Determine the annotation color mapping (either fixed or via callable).
    if callable(annotation_to_color):
        color_map = annotation_to_color(G)
    else:
        color_map = annotation_to_color

    fig, ax = plot_hierarchical_graph(
        G,
        root_node=root_node,
        annotation_to_color=color_map,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=1,
        vert_length=1,
        linewidth=linewidth,
        legend_offset=legend_offset
    )
    fig.set_size_inches(*fig_size)

    # Save the figure with high resolution (600 dpi)
    basename = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{basename}_hierarchical.png"
    # fig.savefig(output_file, dpi=1200) # uncomment to save the figure


def process_duct_system_3d(file_path, annotation_to_color):
    """
    Load a duct system from file_path, create its graph, merge components,
    and plot a 3D view using the provided annotation-to-color mapping.
    """
    duct_systems = load_duct_systems(file_path)
    system_data = select_biggest_duct_system(duct_systems)
    G = create_duct_graph(system_data)
    if len(G.nodes) == 0:
        print("Graph is empty for", file_path)
        return
    merge_disconnected_components(G)
    if callable(annotation_to_color):
        color_map = annotation_to_color(G)
    else:
        color_map = annotation_to_color
    # Plot the 3D system using the Plotly-based function.
    plot_3d_system(G, color_map, label=False)


# Configure each file with its specific parameters.
files_info = [
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\A20.json",
        {"Abnormal": "#FF0000", "Endpoint": "#3689ff"},
        None,
        2,  # linewidth
        -1,  # legend offset
        (7, 5)  # figure size
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\T18-11629.json",
        {"Abnormal": "#FF0000", "Endpoint": "#3689ff"},
        "bp7",
        2,
        -1,
        (3, 6)
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\Coseq_fixed.json",
        {"6": "#39FF14", "8": "#f528f2", "11": "#FFA500", "15": "#FF0000",  "Endpoint": "#3689ff"},
        "bp68",
        2,
        -1,
        (9, 7)
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\Aseq3_4.json",
        {"19": "#39FF14", "23": "#f528f2", "25": "#FFA500", "24": "#FF0000", "Endpoint": "#3689ff"},
        "bp285",
        2,
        -1,
        (8, 6)
    )
]

# Process and plot each duct system in its own figures (both 2D hierarchical and 3D).
for file_path, annotation, root_node, lw, legend_off, fsize in files_info:
    process_duct_system(file_path, annotation, root_node,
                        linewidth=lw, legend_offset=legend_off, fig_size=fsize)
    process_duct_system_3d(file_path, annotation)

plt.show()
