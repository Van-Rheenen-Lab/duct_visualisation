import json
import numpy as np
import networkx as nx
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterio.features import rasterize
from skimage import io
import matplotlib.pyplot as plt

def create_directed_duct_graph(duct_system):
    """
    Creates a directed graph where each segment goes from
    start_bp -> end_bp. This implies start_bp is the 'parent'
    and end_bp is the 'child'.
    """
    G_dir = nx.DiGraph()
    branch_points = duct_system.get("branch_points", {})
    segments = duct_system.get("segments", {})

    for seg_name, seg_data in segments.items():
        start_id = seg_data["start_bp"]
        end_id = seg_data["end_bp"]
        if start_id in branch_points and end_id in branch_points:
            # Add the nodes if missing
            G_dir.add_node(start_id)
            G_dir.add_node(end_id)
            # Directed edge from start -> end
            G_dir.add_edge(start_id, end_id, segment_name=seg_name)

    return G_dir

def get_downstream_subgraph(G_dir, start_node):
    """
    Performs a directed BFS/DFS from 'start_node' and collects
    only the nodes reachable via outgoing edges (successors).
    Returns a subgraph of those downstream nodes (including start_node).
    """
    visited = set()
    frontier = [start_node]
    while frontier:
        current = frontier.pop(0)
        if current not in visited:
            visited.add(current)
            # Move only 'downstream': from current -> child
            for child in G_dir.successors(current):
                if child not in visited:
                    frontier.append(child)
    return G_dir.subgraph(visited).copy()

def remove_downstream_nodes(G_dir, cut_nodes):
    """
    Given a directed graph and one or more 'cut_nodes',
    remove everything downstream (successors) of each cut_node.
    Keeps 'cut_nodes' themselves, but removes all their descendants.
    """
    if isinstance(cut_nodes, str):
        cut_nodes = [cut_nodes]

    to_remove = set()
    for node in cut_nodes:
        # Traverse all successors of 'node' and mark them
        queue = list(G_dir.successors(node))
        while queue:
            current = queue.pop(0)
            if current not in to_remove:
                to_remove.add(current)
                queue.extend(list(G_dir.successors(current)))

    # Now remove them from the graph
    G_dir.remove_nodes_from(to_remove)
    return G_dir

from striping_speedup import (
    load_duct_systems,
    clean_duct_data,
    simplify_duct_system,
    plot_hierarchical_graph_subsegments
)

def plot_downstream_graph_subsegments(
    duct_system,
    start_node,
    duct_mask,
    red_image=None,
    green_image=None,
    yellow_image=None,
    threshold=500,
    draw_nodes=False,
    N=30,
    use_hierarchy_pos=True,
    vert_gap=2,
    orthogonal_edges=True,
    linewidth=2,
    buffer_width=10,
    cut_nodes=None
):
    """
    1) Creates a directed graph from 'duct_system'.
    2) Extracts only the downstream portion from 'start_node'.
    3) (Optionally) removes everything downstream of 'cut_nodes'.
    4) Calls 'plot_hierarchical_graph_subsegments' on the resulting subgraph.
    """

    # Build a directed graph
    G_dir = create_directed_duct_graph(duct_system)

    # Extract just the portion downstream of 'start_node'
    subG = get_downstream_subgraph(G_dir, start_node)

    # If you want to snip off certain parts, do it now
    if cut_nodes is not None:
        remove_downstream_nodes(subG, cut_nodes)

    # Plot
    fig, ax = plot_hierarchical_graph_subsegments(
        subG,
        duct_system,
        root_node=start_node,  # for hierarchy layout
        duct_mask=duct_mask,
        red_image=red_image,
        green_image=green_image,
        yellow_image=yellow_image,
        draw_nodes=draw_nodes,
        threshold=threshold,
        N=N,
        use_hierarchy_pos=use_hierarchy_pos,
        vert_gap=vert_gap,
        orthogonal_edges=orthogonal_edges,
        linewidth=linewidth,
        buffer_width=buffer_width
    )
    return fig, ax


json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
threshold_value = 1000
selected_bp = 'bp330'
cut_nodes = ['bp336']
# cut_nodes = None

if __name__ == "__main__":

    # Load images
    red_image = io.imread(red_image_path) if red_image_path else None
    green_image = io.imread(green_image_path) if green_image_path else None
    yellow_image = io.imread(yellow_image_path) if yellow_image_path else None

    # Load the duct system
    duct_systems = load_duct_systems(json_path)
    system_idx = 1
    duct_system = duct_systems[system_idx]

    # Clean/simplify
    if len(duct_system["segments"]) > 0:
        duct_system = clean_duct_data(duct_system)
        first_branch_node = list(duct_system["branch_points"].keys())[0]
        duct_system = simplify_duct_system(duct_system, first_branch_node)

    # Build a polygon mask of the duct
    with open(duct_borders_path, 'r') as f:
        duct_borders = json.load(f)
    duct_polygon = unary_union([shape(feat['geometry']) for feat in duct_borders['features']])
    base_shape = red_image.shape
    shapes = [(duct_polygon, 1)]
    duct_mask = rasterize(shapes, out_shape=base_shape, fill=0, dtype=np.uint8, all_touched=False)

    plot_downstream_graph_subsegments(
        duct_system=duct_system,
        start_node=selected_bp,       # The specific branch node we want to start from
        duct_mask=duct_mask,
        red_image=red_image,
        green_image=green_image,
        yellow_image=yellow_image,
        threshold=threshold_value,
        draw_nodes=True,
        N=20,
        use_hierarchy_pos=True,
        vert_gap=2,
        orthogonal_edges=True,
        linewidth=1.5,
        buffer_width=10,
        cut_nodes=cut_nodes # pass in one or more nodes to snip
    )

    # save figure at high resolution

    plt.savefig('3_colors_frombp330_cutbp336.png', dpi=300)

    plt.show()