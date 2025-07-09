import os
import matplotlib.pyplot as plt
import networkx as nx
from utils.loading_saving import load_duct_systems, create_duct_graph, select_biggest_duct_system, create_directed_duct_graph, find_root
from utils.plotting_trees import plot_hierarchical_graph
from utils.fixing_annotations import connect_component_to_main, simplify_graph, orient_graph_from_root
from utils.plotting_3d import plot_3d_system

"""
This script plots duct systems as dendrograms, but for endpoints (leaves) that do not have a TDLU or other annotation, it draws a dashed black line (---) extending from them, indicating the duct continued outside the imaging area. TDLUs are not specially labeled.
"""

def merge_disconnected_components(G, coord_tol=10):
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

def plot_hierarchical_graph_with_unannotated_extensions(
        G,
        root_node=None,
        use_hierarchy_pos=True,
        vert_gap=1,
        orthogonal_edges=True,
        vert_length=1,
        annotation_to_color=None,
        segment_color_map=None,
        linewidth=2,
        legend_offset=-1,
        font_size=12,
        extension_length=1.4,
        fig_size=(7, 5)
):
    fig, ax = plot_hierarchical_graph(
        G,
        root_node=root_node,
        use_hierarchy_pos=use_hierarchy_pos,
        vert_gap=vert_gap,
        orthogonal_edges=orthogonal_edges,
        vert_length=vert_length,
        annotation_to_color=annotation_to_color,
        segment_color_map=segment_color_map,
        linewidth=linewidth,
        legend_offset=legend_offset,
        font_size=font_size
    )
    # Get node positions
    if use_hierarchy_pos:
        from utils.plotting_trees import hierarchy_pos
        pos = hierarchy_pos(G, root=root_node, vert_gap=vert_gap)
    else:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB', root=root_node)

    # Find the highest y-value (top of the tree)
    highest_y = max(y for x, y in pos.values())
    # For each endpoint, check if it is unannotated and add a dotted extension if so
    for node in G.nodes:
        # Only consider endpoints that are not the root node and not at the top of the tree
        if node == root_node:
            continue
        if node not in pos:
            continue
        x, y = pos[node]
        if y == highest_y:
            continue
        if G.degree(node) == 1:
            neighbor = list(G.neighbors(node))[0]
            edge_data = G.get_edge_data(node, neighbor)
            properties = edge_data.get('properties', {})
            ann = properties.get('Annotation', None)
            if isinstance(ann, list):
                ann = ann[0] if ann else None
            if not ann:
                # Draw a very fine dotted line straight down from the endpoint (no offset, starts at endpoint)
                ax.plot([x, x], [y, y - extension_length], color='black', linewidth=2, linestyle=(0, (0.5, 0.4)), zorder=2)
    return fig, ax


def process_duct_system_with_extensions(
        file_path, annotation_to_color, root_node=None,
        linewidth=2, legend_offset=-1, fig_size=(7, 5), simplify=True,
        coord_tol=10                     # <- make tolerance configurable
):
    duct_systems = load_duct_systems(file_path)
    system_data  = select_biggest_duct_system(duct_systems)

    G_dir = create_directed_duct_graph(system_data)

    if root_node is None:
        root_node = find_root(G_dir)

    G_tmp = nx.Graph(G_dir)                     # undirected copy
    merge_disconnected_components(G_tmp, coord_tol=coord_tol)
    G_dir = nx.DiGraph(G_tmp)                   # bring bridges back


    G_dir = orient_graph_from_root(G_dir, root_node)

    if simplify:
        print_graph_degrees(G_dir)
        G_dir = simplify_graph(G_dir, main_branch_node=root_node)

    G_plot = nx.Graph(G_dir)

    color_map = annotation_to_color(G_plot) if callable(annotation_to_color) else annotation_to_color

    fig, ax = plot_hierarchical_graph_with_unannotated_extensions(
        G_plot,
        root_node=root_node,
        annotation_to_color=color_map,
        linewidth=linewidth,
        legend_offset=legend_offset,
        fig_size=fig_size
    )
    fig.set_size_inches(*fig_size)
    # save as svg
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    fig.savefig(f"{file_name}_extensions.svg", format='svg')



def process_duct_system_3d(file_path, annotation_to_color):
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
    plot_3d_system(G, color_map, label=False)

def print_graph_degrees(G):
    print("[Graph Degrees]")
    for node in G.nodes():
        print(f"{node}: in={G.in_degree(node)}, out={G.out_degree(node)}")
    print("[Edges]")
    for u, v in G.edges():
        print(f"{u} -> {v}")

files_info = [
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\A20.json",
        {"Abnormal": "#FF0000", "Endpoint": "#3689FF"},
        None,
        2,
        -1,
        (13, 10),
        True
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\T18-11629.json",
        {"Abnormal": "#FF0000", "Endpoint": "#3689FF"},
        "bp7",
        2,
        -1,
        (3, 6),
        True
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\Coseq_fixed.json",
        {"6": "#73BF43", "8": "#f528f2", "11": "#FFA500", "15": "#FF0000",  "Endpoint": "#000000", "1": "#000000", "2": "#000000", "3": "#000000", "4": "#000000", "5": "#000000", "7": "#000000", "9": "#000000", "10": "#000000", "12": "#000000", "13": "#000000", "14": "#000000", "16": "#000000"},
        "bp68",
        2,
        -3,
        (10, 7),
        True
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations example hendrik\Aseq3_4.json",
        {"Endpoint": "#000000", "TDLU": "#000000", "13": "#000000", "15": "#000000", "19": "#73BF43", "20": "#000000", "21": "#000000", "22": "#000000", "23": "#f528f2", "24": "#FF0000", "25": "#FFA500", "26": "#000000"},
        "bp285",
        2,
        -3,
        (8, 6),
        True
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations hendrik\A15_use-bp298-as-origin.json",
        {"DCIS": "#FF0000","Lesion": "#FF0000", "TDLU": "#0080FE"},
        "bp298",
        2,
        -0.6,
        (18, 14),
        True
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations hendrik\T19-11890.json",
        {"DCIS": "#FF0000","Lesion": "#FF0000", "TDLU": "#0080FE"},
        None,
        2,
        -0.8,
        (4, 5),
        True
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations hendrik\T18-07933.json",
        {"DCIS": "#FF0000","Lesion": "#FF0000", "TDLU": "#0080FE"},
        None,
        2,
        -1,
        (2, 6),
        True
    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations hendrik\A18.json",
        {"Abnormal": "#FF0000", "Endpoint": "#3689FF"},
        None,
        2,
        -1,
        (18, 14),
        True

    ),
    (
        r"I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\duct annotations hendrik\T18-05203.json",
        {"Abnormal": "#FF0000", "Endpoint": "#3689FF"},
        None,
        2,
        -1,
        (4, 5),
        False

    )

]

for file_path, annotation, root_node, lw, legend_off, fsize, simplify in files_info:
    process_duct_system_with_extensions(file_path, annotation, root_node,
                        linewidth=lw, legend_offset=legend_off, fig_size=fsize, simplify=simplify)
    process_duct_system_3d(file_path, annotation)

plt.show() 