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
        initial_termination_prob = 0.1
    )

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

    # Create a color map based on any annotations
    color_map = create_annotation_color_map(system_data, colormap_name='tab10')

    # 3) Plot the tree with your hierarchical graph function
    fig, ax = plot_hierarchical_graph(
        G,
        system_data=system_data,
        annotation_to_color=color_map,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=5
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

    plt.show()
