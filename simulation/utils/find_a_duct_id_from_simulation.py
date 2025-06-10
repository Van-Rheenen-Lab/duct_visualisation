import random
import matplotlib.pyplot as plt
from simulation.puberty import simulate_ductal_tree
from analysis.utils.plotting_trees import hierarchy_pos, plot_hierarchical_graph

# Global variables for interactive highlighting
selected_artist = None
selected_annotation = None


def on_click(event, pos, ax, fig, G):
    """
    When a click event occurs, this function finds the nearest node
    (duct) from the computed hierarchical positions, prints its ID,
    and highlights it on the plot.
    """
    global selected_artist, selected_annotation

    # Ensure the click is within the axes
    if event.xdata is None or event.ydata is None:
        return

    click_point = (event.xdata, event.ydata)
    min_distance = float('inf')
    closest_node = None
    for node, (x, y) in pos.items():
        distance = ((click_point[0] - x) ** 2 + (click_point[1] - y) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_node = node

    if closest_node is not None:
        print(f"Selected duct id: {closest_node}")
        # Remove previous highlights if any.
        if selected_artist is not None:
            selected_artist.remove()
        if selected_annotation is not None:
            selected_annotation.remove()
        # Highlight the selected node with a red marker and annotate it.
        x, y = pos[closest_node]
        selected_artist = ax.scatter([x], [y], s=300, c='red', zorder=10)
        selected_annotation = ax.annotate(f"ID: {closest_node}", (x, y),
                                          textcoords="offset points", xytext=(10, 10),
                                          color='red', fontsize=12)
        fig.canvas.draw()


def main():
    # Set the random seed as requested.
    random.seed(42)

    # --- Simulation parameters ---
    n_clones = 170
    bifurcation_prob = 0.01
    initial_termination_prob = 0.25
    max_cells = 6000000

    # Simulate the pubertal ductal tree.
    G_puberty, _ = simulate_ductal_tree(
        max_cells=max_cells,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=n_clones / 2,
        initial_center_count=n_clones / 2,
        initial_termination_prob=initial_termination_prob
    )

    # Choose the root node (here, simply the first node in the graph).
    root_node = list(G_puberty.nodes())[0]

    # Plot the ductal tree using your hierarchical plotting function.
    # We are not adding any clone annotations here.
    fig, ax = plot_hierarchical_graph(
        G_puberty,
        root_node=root_node,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        annotation_to_color=None,  # No annotation mapping.
        segment_color_map=None,
        linewidth=1,
        legend_offset=-0.1
    )
    ax.set_title("Pubertal Ductal Tree\n(Click a duct to highlight its ID)")

    # Compute the hierarchical positions using your provided function.
    # Ensure that the vert_gap here matches that used in the plotting function.
    pos = hierarchy_pos(G_puberty, root=root_node, vert_gap=1)

    # Connect the click event to our on_click handler.
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, pos, ax, fig, G_puberty))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
