from analysis.utils.plotting_trees import plot_hierarchical_graph, create_annotation_color_map
import matplotlib.pyplot as plt

def plotting_ducts(G, vert_gap=2, color_map=None, root_node=None, clone_attr="duct_clones"):
    """
    Plots the hierarchical ductal tree. It annotates each segment based on the
    clones (using the attribute given by clone_attr) present on that edge.
    """
    system_data = {"segments": {}}

    # Loop over edges, reading out the clones from the specified attribute.
    for (u, v) in G.edges():
        segment_name = f"duct_{u}_to_{v}"
        clones = G[u][v].get(clone_attr, [])
        ann = None
        if color_map:
            for annotations in color_map:
                if annotations in [str(x) for x in clones]: # bit hacky but we have to change the clones to string
                    ann = annotations

        if ann:
            system_data["segments"][segment_name] = {"properties": {"Annotation": ann}}
        else:
            system_data["segments"][segment_name] = {"properties": {}}

        # Store segment name for reference
        G[u][v]["segment_name"] = segment_name

    if not color_map:
        color_map = create_annotation_color_map(system_data, colormap_name='tab10')

    fig, ax = plot_hierarchical_graph(
        G,
        system_data=system_data,
        annotation_to_color=color_map,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=vert_gap,
        linewidth=0.9,
        root_node=root_node
    )

    # Calculate and print the average number of cells per segment
    total_cells = 0
    edge_count = 0
    for (u, v) in G.edges():
        edge_clones = G[u][v].get(clone_attr, [])
        total_cells += len(edge_clones)
        edge_count += 1

    avg_cells = total_cells / edge_count if edge_count > 0 else 0
    print(f"Average number of cells per segment: {avg_cells}")

    plt.tight_layout()
    return fig, ax


def plot_selected_ducts(G, selected_ducts, vert_gap=2, color_map=None):
    """
    Plots a hierarchical ductal tree highlighting segments whose child node
    is in the list `selected_ducts`. Each highlighted segment is annotated
    with its node ID.
    """
    system_data = {"segments": {}}

    for (parent, child) in G.edges():
        segment_name = f"duct_{parent}_to_{child}"
        G[parent][child]["segment_name"] = segment_name
        if child in selected_ducts:
            system_data["segments"][segment_name] = {"properties": {"Annotation": str(child)}}
        else:
            system_data["segments"][segment_name] = {"properties": {}}

    if not color_map:
        color_map = create_annotation_color_map(
            system_data,
            fixed_annotation=None,
            colormap_name="tab10"
        )

    fig, ax = plot_hierarchical_graph(
        G,
        system_data=system_data,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=vert_gap,
        annotation_to_color=color_map,
        linewidth=0.9
    )

    plt.title("Selected Ducts Highlighted by Duct ID")
    plt.tight_layout()
    return fig, ax
