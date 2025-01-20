import json
import numpy as np
import networkx as nx
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterio.features import rasterize
from skimage import io
import matplotlib.pyplot as plt
from shapely.validation import make_valid
from analysis.utils.loading_saving import load_duct_systems, create_duct_graph
from striping_speedup import (
    load_duct_systems,
    clean_duct_data,
    simplify_duct_system,
    plot_hierarchical_graph_subsegments
)

import networkx as nx
import warnings


def create_directed_duct_graph(duct_system):
    """
    Creates a directed graph with nodes for all branch_points,
    and directed edges from start_bp -> end_bp for each segment.

    If a segment references an undefined branch point, logs a warning
    and skips that segment.
    """
    G_dir = nx.DiGraph()
    branch_points = duct_system.get("branch_points", {})
    segments = duct_system.get("segments", {})

    # Add all branch points as nodes with attributes
    for bp_name, bp_data in branch_points.items():
        G_dir.add_node(bp_name, **bp_data)

    # Add directed edges for valid segments
    for seg_name, seg_data in segments.items():
        start_bp = seg_data["start_bp"]
        end_bp = seg_data["end_bp"]

        if start_bp not in branch_points or end_bp not in branch_points:
            warnings.warn(f"Segment '{seg_name}' references undefined branch points.")
            continue

        G_dir.add_edge(start_bp, end_bp, segment_name=seg_name)

    return G_dir


import networkx as nx
from collections import deque

def get_downstream_subgraph(G_dir, start_node):
    """
    Performs a directed BFS from 'start_node' and collects
    all nodes reachable via outgoing edges (successors).
    Returns a new subgraph containing those downstream nodes
    (including start_node).

    Parameters
    ----------
    G_dir : nx.DiGraph
        A directed graph.
    start_node : hashable
        The node in G_dir from which to begin the BFS.

    Returns
    -------
    nx.DiGraph
        A subgraph of G_dir containing all nodes reachable
        via successors of `start_node` (plus `start_node`),
        along with edges among those nodes.
    """
    # If the start_node doesn't exist in the graph, return an empty subgraph
    if start_node not in G_dir:
        return G_dir.subgraph([]).copy()

    visited = set()
    queue = deque([start_node])

    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            # Only explore children in the "downstream" direction
            for child in G_dir.successors(current):
                if child not in visited:
                    queue.append(child)

    # Return a subgraph copy containing visited nodes (and edges among them)
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

    if start_node is None:
        first_bp = list(G_dir.nodes)[0]
        # find parent of first branch point iteratively
        while len(list(G_dir.predecessors(first_bp))) == 1:
            first_bp = list(G_dir.predecessors(first_bp))[0]
        start_node = first_bp
        print(f"Starting from first branch point: {start_node}")

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


# json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
# duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'
#
# green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
# yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
# red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
# threshold_value = 1000
# selected_bp = 'bp330'
# cut_nodes = ['bp336']
# system_idx = 1

red_image_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\28052024_2435322_L5_ecad_mAX-0006.tif'
duct_borders_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\annotations_exported.geojson'
json_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'
green_image_path = None
yellow_image_path = None
threshold_value = 500
selected_bp = None
cut_nodes = None
system_idx = 0

if __name__ == "__main__":

    # Load images
    red_image = io.imread(red_image_path) if red_image_path else None
    green_image = io.imread(green_image_path) if green_image_path else None
    yellow_image = io.imread(yellow_image_path) if yellow_image_path else None

    # Load the duct system
    duct_systems = load_duct_systems(json_path)
    duct_system = duct_systems[system_idx]

    # Clean/simplify
    if len(duct_system["segments"]) > 0:
        duct_system = clean_duct_data(duct_system)
        first_branch_node = list(duct_system["branch_points"].keys())[0]
        duct_system = simplify_duct_system(duct_system, first_branch_node)

    # Build a polygon mask of the duct
    with open(duct_borders_path, 'r') as f:
        duct_borders = json.load(f)

    valid_geoms = []
    for feat in duct_borders['features']:
        geom = shape(feat['geometry'])
        if not geom.is_valid:

            geom = make_valid(geom)

        if geom.is_valid:
            valid_geoms.append(geom)
        else:
            print("Skipping geometry that could not be fixed:", feat['geometry'])

    duct_polygon = unary_union(valid_geoms)
    base_shape = red_image.shape
    shapes = [(duct_polygon, 1)]
    duct_mask = rasterize(shapes, out_shape=base_shape, fill=0, dtype=np.uint8, all_touched=False)

    # plot_hierarchical_graph_subsegments(G,
    #                                     duct_system,
    #                                     root_node=selected_bp,
    #                                     duct_mask=duct_mask,
    #                                     red_image=red_image,
    #                                     green_image=green_image,
    #                                     yellow_image=yellow_image,
    #                                     threshold=threshold_value,
    #                                     N=30,
    #                                     use_hierarchy_pos=True,
    #                                     vert_gap=5,
    #                                     orthogonal_edges=True,
    #                                     linewidth=1,
    #                                     buffer_width=10
    # )

    plot_downstream_graph_subsegments(
        duct_system=duct_system,
        start_node=selected_bp,       # The specific branch node we want to start from
        duct_mask=duct_mask,
        red_image=red_image,
        green_image=green_image,
        yellow_image=yellow_image,
        threshold=threshold_value,
        draw_nodes=False,
        N=30,
        use_hierarchy_pos=True,
        vert_gap=5,
        orthogonal_edges=True,
        linewidth=1,
        buffer_width=10,
        cut_nodes=cut_nodes # pass in one or more nodes to snip
    )

    # save figure at high resolution

    plt.savefig('3_colors_frombp330_cutbp336.png', dpi=300)

    plt.show()