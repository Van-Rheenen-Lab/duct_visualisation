from skimage import io
import matplotlib.pyplot as plt
from collections import deque
from analysis.utils.loading_saving import load_duct_systems, create_directed_duct_graph, find_root, load_duct_mask
from analysis.utils.fixing_annotations import simplify_graph
from analysis.utils.plotting_striped_trees import plot_hierarchical_graph_subsegments


def get_downstream_subgraph(G, start_node):
    """
    Performs a directed traversal from `start_node` and collects all nodes reachable via successors.
    Returns a new subgraph containing those downstream nodes (including `start_node`).
    """
    if start_node not in G:
        return G.subgraph([]).copy()
    visited = set()
    queue = deque([start_node])
    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            for child in G.successors(current):
                if child not in visited:
                    queue.append(child)
    return G.subgraph(visited).copy()


def remove_downstream_nodes(G, cut_nodes):
    """
    Given a directed graph and one or more 'cut_nodes', remove everything downstream (all successors)
    of each cut_node. The cut_nodes themselves remain in the graph.
    """
    if isinstance(cut_nodes, str):
        cut_nodes = [cut_nodes]
    to_remove = set()
    for node in cut_nodes:
        queue = list(G.successors(node))
        while queue:
            current = queue.pop(0)
            if current not in to_remove:
                to_remove.add(current)
                queue.extend(list(G.successors(current)))
    G.remove_nodes_from(to_remove)
    return G


def plot_downstream_graph_subsegments(
        G,
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
        cut_nodes=None,
        root=None
):
    """
    1) Optionally simplifies the graph.
    2) Finds a root (via find_root) and extracts the downstream subgraph from it.
    3) Optionally removes nodes downstream of given cut_nodes.
    4) Calls 'plot_hierarchical_graph_subsegments' on the resulting subgraph.

    All duct system metadata is assumed to be stored in the graph G.
    """
    # Simplify the graph (if needed)
    G_simpl = simplify_graph(G)
    # Determine the root node using the updated find_root (which works solely on G)
    if root is None:
        root = find_root(G_simpl)

    subG = get_downstream_subgraph(G_simpl, root)
    # Optionally remove unwanted downstream nodes.
    if cut_nodes is not None:
        remove_downstream_nodes(subG, cut_nodes)
    # Plot the subgraph with subsegment stripes.
    fig, ax = plot_hierarchical_graph_subsegments(
        subG,
        root_node=root,
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


if __name__ == "__main__":
    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

    green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
    yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
    red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
    threshold_value = 400
    system_idx = 1
    cut_nodes = ["bp333"]
    root = "bp329"
    n_subsegments = 30

    # Load images
    red_image = io.imread(red_image_path) if red_image_path else None
    green_image = io.imread(green_image_path) if green_image_path else None
    yellow_image = io.imread(yellow_image_path) if yellow_image_path else None

    # Load duct systems from JSON and select one.
    duct_systems = load_duct_systems(json_path)
    duct_system = duct_systems[system_idx]

    # Create a directed graph from the duct system.
    G = create_directed_duct_graph(duct_system)

    duct_polygon, duct_mask = load_duct_mask(duct_borders_path, out_shape=red_image.shape)

    # Plot the downstream subgraph with striped subsegments.
    fig, ax = plot_downstream_graph_subsegments(
        G,
        duct_mask=duct_mask,
        red_image=red_image,
        green_image=green_image,
        yellow_image=yellow_image,
        threshold=threshold_value,
        draw_nodes=False,
        N=n_subsegments,
        use_hierarchy_pos=True,
        vert_gap=2,
        orthogonal_edges=True,
        linewidth=1.2,
        buffer_width=10,
        cut_nodes=cut_nodes,
        root=root
    )

    # plt.savefig('2473536_Cft_24W_thresh1000_nostripes.png', dpi=800)
    plt.show()
