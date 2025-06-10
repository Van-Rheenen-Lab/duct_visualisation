import matplotlib.pyplot as plt
from analysis.utils.plotting_trees import plot_hierarchical_graph, create_annotation_color_map


def plotting_ducts(G, vert_gap=2, color_map=None, root_node=None, clone_attr="duct_clones"):
    """
    Plots the hierarchical ductal tree. It annotates each segment based on the
    clones (using the attribute given by clone_attr) present on that edge.

    This version updates each edge’s properties directly in the graph.
    """
    # Loop over edges and update the edge attributes.
    for (u, v) in G.edges():
        segment_name = f"duct_{u}_to_{v}"
        # Set the segment name.
        G[u][v]["segment_name"] = segment_name

        # Read clones and determine an annotation if one of the clone IDs (as string)
        # is present in the provided color_map keys.
        clones = G[u][v].get(clone_attr, [])
        ann = None
        if color_map:
            # Compare each key (annotation) in color_map to clones (converted to string)
            for key in color_map.keys():
                if key in [str(x) for x in clones]:
                    ann = key
                    break

        # Store the annotation (or an empty string if none) in the edge properties.
        G[u][v]["properties"] = {"Annotation": ann} if ann is not None else {"Annotation": ""}

    # If no color_map is provided, generate one from the graph's edge properties.
    if not color_map:
        color_map = create_annotation_color_map(G, colormap_name='tab10')

    # Plot using the hierarchical graph function that now uses only the graph.
    fig, ax = plot_hierarchical_graph(
        G,
        root_node=root_node,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=vert_gap,
        annotation_to_color=color_map,
        linewidth=0.9
    )

    # Calculate and print the average number of cells per segment.
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

def plot_selected_ducts(G,
                        selected_ducts,
                        vert_gap=2,
                        color_map=None,
                        root_node=None,
                        linewidth = 0.9
                        ):
    """
    Plots a hierarchical ductal tree highlighting segments whose child node is
    in `selected_ducts`.  Each highlighted segment is annotated with that node ID.
    """
    for parent, child in G.edges():
        seg_name = f"duct_{parent}_to_{child}"
        G[parent][child]["segment_name"] = seg_name
        G[parent][child]["properties"] = (
            {"Annotation": str(child)} if child in selected_ducts
            else {"Annotation": ""}
        )

    if not color_map:
        color_map = create_annotation_color_map(G,
                                                fixed_annotation=None,
                                                colormap_name="tab10")

    fig, ax = plot_hierarchical_graph(
        G,
        root_node=root_node,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=vert_gap,
        annotation_to_color=color_map,
        linewidth=linewidth
    )

    plt.title("Selected Ducts Highlighted by Duct ID")
    plt.tight_layout()
    return fig, ax
