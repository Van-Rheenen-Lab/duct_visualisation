from analysis.utils.plotting_trees import plot_hierarchical_graph, create_annotation_color_map

def plotting_ducts(G, vert_gap=2, color_map=None):

    # 2) Build the system_data structure
    system_data = {"segments": {}}

    # Loop over edges, read out duct_clones to decide how to annotate
    for (u, v) in G.edges():
        segment_name = f"duct_{u}_to_{v}"
        duct_clones = G[u][v].get("duct_clones", [])

        # Optionally set an annotation if certain clones appear
        ann = None
        if 42 in duct_clones:
            ann = "42"
        elif 84 in duct_clones:
            ann = "90"
        elif 12 in duct_clones:
            ann = "12"
        elif 24 in duct_clones:
            ann = "24"

        if ann:
            system_data["segments"][segment_name] = {
                "properties": {"Annotation": ann}
            }
        else:
            system_data["segments"][segment_name] = {
                "properties": {}
            }

        # Store segment name for reference
        G[u][v]["segment_name"] = segment_name

    if not color_map:
        # Create a color map based on any annotations
        color_map = create_annotation_color_map(system_data, colormap_name='tab10')

    # 3) Plot the tree with your hierarchical graph function
    fig, ax = plot_hierarchical_graph(
        G,
        system_data=system_data,
        annotation_to_color=color_map,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap= vert_gap,
        linewidth=1.5
    )

    # 4) Calculate the average number of cells per segment
    total_cells = 0
    edge_count = 0

    for (u, v) in G.edges():
        edge_clones = G[u][v].get("duct_clones", [])
        total_cells += len(edge_clones)
        edge_count += 1

    if edge_count > 0:
        avg_cells = total_cells / edge_count
    else:
        avg_cells = 0

    print(f"Average number of cells per segment: {avg_cells}")

import matplotlib.pyplot as plt
from analysis.utils.plotting_trees import (
    plot_hierarchical_graph,
    create_annotation_color_map
)

def plot_selected_ducts(G, selected_ducts, vert_gap=2, color_map=None):
    """
    Plots a hierarchical ductal tree, highlighting segments whose child node
    is in the list `selected_ducts`. Each selected duct gets its own annotation
    named after that duct's node ID (e.g., "2", "10", etc.), rather than
    labeling them all "Selected."

    Parameters
    ----------
    G : networkx.DiGraph
        The ductal tree graph. Each edge can have additional attributes
        like 'duct_clones' etc.
    selected_ducts : list
        List of node IDs to highlight/annotate.
    vert_gap : float, default=2
        Vertical gap between levels in the hierarchy plot.
    color_map : dict, optional
        A mapping of annotation -> color. If not provided, a color map is created
        automatically that assigns unique colors to each duct ID.
    """

    # Build a system_data dictionary (as used in your pipeline) so we can
    # store annotation properties on each segment.
    system_data = {"segments": {}}

    # Loop over the edges, mark edges that go to a selected child node
    for (parent, child) in G.edges():
        segment_name = f"duct_{parent}_to_{child}"
        # Store a reference to the segment name in the graph
        G[parent][child]["segment_name"] = segment_name

        # If the child is one of the selected ducts, annotate with its node ID
        if child in selected_ducts:
            system_data["segments"][segment_name] = {
                "properties": {"Annotation": str(child)}
            }
        else:
            system_data["segments"][segment_name] = {
                "properties": {}
            }

    # If no color_map is supplied, create one that assigns distinct colors
    # for each annotation found in system_data. This ensures each selected
    # duct ID is highlighted uniquely.
    if not color_map:
        color_map = create_annotation_color_map(
            system_data,
            fixed_annotation=None,  # Turn off special handling for a single label
            colormap_name="tab10"
        )

    # Now plot using your existing hierarchical function
    fig, ax = plot_hierarchical_graph(
        G,
        system_data=system_data,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=vert_gap,
        annotation_to_color=color_map,
        linewidth=1.5
    )

    plt.title("Selected Ducts Highlighted by Duct ID")


    return fig, ax
