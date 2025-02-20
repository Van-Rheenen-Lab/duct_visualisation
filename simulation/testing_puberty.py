import random
import numpy as np
import matplotlib.pyplot as plt
from puberty import simulate_ductal_tree
from analysis.utils.plotting_trees import plot_hierarchical_graph, create_annotation_color_map

if __name__ == "__main__":
    random.seed(40)

    G, _ = simulate_ductal_tree(
        max_cells=3000000,
        bifurcation_prob=0.01,
        initial_side_count=50,
        initial_center_count=50,
        initial_termination_prob=0.1
    )

    # Loop over edges to update metadata directly in the graph.
    for (u, v) in G.edges():
        segment_name = f"duct_{u}_to_{v}"
        duct_clones = G[u][v].get("duct_clones", [])

        # Determine annotation based on the presence of certain clones.
        ann = None
        if 42 in duct_clones:
            ann = "42"
        elif 84 in duct_clones:
            ann = "90"
        elif 12 in duct_clones:
            ann = "12"
        elif 24 in duct_clones:
            ann = "24"

        # Store the segment name and annotation (if any) in the edge attributes.
        G[u][v]["segment_name"] = segment_name
        if ann:
            G[u][v]["properties"] = {"Annotation": ann}
        else:
            G[u][v]["properties"] = {}

    # Create a color map for annotations directly from the graph.
    color_map = create_annotation_color_map(G, colormap_name='tab10')

    # Plot the ductal tree using the hierarchical graph function.
    fig, ax = plot_hierarchical_graph(
        G,
        annotation_to_color=color_map,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=5
    )

    # Calculate the average number of cells per segment.
    total_cells = 0
    edge_count = 0
    for (u, v) in G.edges():
        edge_clones = G[u][v].get("duct_clones", [])
        total_cells += len(edge_clones)
        edge_count += 1

    avg_cells = total_cells / edge_count if edge_count > 0 else 0
    print(f"Average number of cells per segment: {avg_cells}")

    plt.show()
