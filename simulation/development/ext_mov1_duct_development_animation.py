import networkx as nx
import matplotlib.pyplot as plt
import random
import os
import imageio.v2 as imageio
from simulation.puberty_deposit_elimination import simulate_ductal_tree
from analysis.utils.plotting_trees import (
    plot_hierarchical_graph,
    create_annotation_color_map
)

def create_duct_growth_gif_with_annotations(
        max_cells=3_000_000,
        bifurcation_prob=0.01,
        initial_side_count=170/2,
        initial_center_count=170/2,
        initial_termination_prob=0.25,
        final_termination_prob=0.55,
        output_filename="duct_growth.gif"
):
    """
    Runs the puberty simulation, then creates a GIF showing the network growing
    (one frame per newly created duct). The edges are colored by annotations
    using your existing `plot_hierarchical_graph`.
    """
    random.seed(41)

    # 1) Run the simulation
    G_final, progress_data = simulate_ductal_tree(
        max_cells=max_cells,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=initial_side_count,
        initial_center_count=initial_center_count,
        initial_termination_prob=initial_termination_prob,
        final_termination_prob=final_termination_prob
    )

    # 2) Get creation history: (parent_node, child_node, iteration_when_created)
    duct_events = progress_data["duct_creation_history"]
    duct_events_sorted = sorted(duct_events, key=lambda x: x[2])

    # 3) Build incremental snapshots
    G_empty = nx.DiGraph()
    # Copy node attributes so the hierarchy layout won't break
    # (your function uses a root node and descriptions, etc.)
    G_empty.add_nodes_from(G_final.nodes(data=True))

    snapshots = []
    current_graph = nx.DiGraph(G_empty)

    for (parent, child, iteration_formed) in duct_events_sorted:
        if G_final.has_edge(parent, child):
            edge_attrs = G_final[parent][child]
            current_graph.add_edge(parent, child, **edge_attrs)
            current_graph[parent][child]["segment_name"] = f"duct_{parent}_to_{child}"

        snapshots.append((nx.DiGraph(current_graph), iteration_formed))

    # Create a 'frames' directory for storing PNGs
    if not os.path.exists("frames"):
        os.makedirs("frames")

    filenames = []  # to store each PNG name

    for i, (G_snap, iter_label) in enumerate(snapshots):
        # No annotation or color logic; just plot the graph as-is
        for (u, v) in G_snap.edges():
            G_snap[u][v]["segment_name"] = f"duct_{u}_to_{v}"

        # Plot this snapshot with hierarchical layout (default color)
        fig, ax = plot_hierarchical_graph(
            G_snap,
            root_node=0,
            use_hierarchy_pos=True,
            vert_gap=2,
            orthogonal_edges=True,
            linewidth=0.6
        )

        # Save to PNG (fix shape by removing 'tight' and using a consistent fig size)
        frame_filename = os.path.join("frames", f"frame_{i:03d}.png")
        fig.set_size_inches(6, 5)       # Ensure same figure size each time
        fig.savefig(frame_filename, bbox_inches=None)
        plt.close(fig)
        filenames.append(frame_filename)

    # 5) Combine PNGs into a GIF using imageio.v2
    images = []
    for fn in filenames:
        img = imageio.imread(fn)  # consistent shape
        images.append(img)

    # duration=0.8 means 0.8s per frame
    imageio.mimsave(output_filename, images, duration=0.03)

    print(f"GIF saved to {output_filename}")

if __name__ == "__main__":
    create_duct_growth_gif_with_annotations()
