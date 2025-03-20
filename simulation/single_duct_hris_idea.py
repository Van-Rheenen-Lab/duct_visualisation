import copy
import random
import matplotlib.pyplot as plt
import colorcet as cc  # for Glasbey palette
import numpy as np

# Import your simulation functions.
from puberty import simulate_ductal_tree
from adulthood import simulate_adulthood


def gather_clones_for_duct(G, duct, puberty_clones=True):
    """
    Given a graph G and a duct node ID, returns the list of clone IDs
    from the incoming edge to that duct.
    For pubertal clones, returns the "duct_clones" attribute.
    For adult clones, returns the "adult_clones" attribute.
    """
    parents = list(G.predecessors(duct))
    if not parents:
        return []
    parent = parents[0]
    if puberty_clones:
        return G[parent][duct].get("duct_clones", [])
    else:
        return G[parent][duct].get("adult_clones", [])


def extract_unique_clone_order(clone_list):
    """
    Returns a list of unique clone IDs, preserving the order of first appearance.
    """
    unique = []
    seen = set()
    for cid in clone_list:
        if cid not in seen:
            unique.append(cid)
            seen.add(cid)
    return unique


def plot_ductal_clone_line_at_x(clone_ids, ax, clone_color_map, x=0, line_width=14, default_color="black"):
    """
    Plots a vertical line for the provided clone IDs at horizontal position x.
    The normalized duct length is represented on the y-axis (from 1 at the bottom to 0 at the top).
    The clones are divided equally along this length.
    """
    n = len(clone_ids)
    if n == 0:
        return
    cell_height = 1.0 / n
    # Compute positions so that the first clone appears at y = 1 and the last near y = 0.
    y_positions = [1 - i * cell_height for i in range(n + 1)]
    for i in range(n - 1):
        cid = str(clone_ids[i])
        color = clone_color_map[cid] if cid in clone_color_map else default_color
        ax.vlines(x=x, ymin=y_positions[i+1], ymax=y_positions[i],
                  colors=color, linewidth=line_width)


def main():
    random.seed(42)

    # --- Run the Puberty Simulation ---
    print("Simulating puberty...")
    max_cells = 6_000_000
    bifurcation_prob = 0.01
    initial_termination_prob = 0.25
    initial_side_count = 85
    initial_center_count = 85

    G_puberty, _ = simulate_ductal_tree(
        max_cells=max_cells,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=initial_side_count,
        initial_center_count=initial_center_count,
        initial_termination_prob=initial_termination_prob
    )
    # Save the pubertal state for adulthood simulation.
    G_puberty_orig = copy.deepcopy(G_puberty)

    duct_id = 183  # duct to visualize
    n_iterations = 33  # number of adulthood iterations

    # --- Run the Adulthood Simulation Once ---
    print("Simulating adulthood...")
    # simulate_adulthood returns (G_adulthood, progress_data_adult, round_graphs)
    G_adulthood, progress_data_adult, round_graphs = simulate_adulthood(
        copy.deepcopy(G_puberty_orig),
        rounds=n_iterations,
        output_graphs=True,
        seed=42
    )

    # --- Collect Clones for the Single Duct ---
    # For pubertal clones:
    pubertal_clones_over_iterations = []
    # First, add the original pubertal state.
    pubertal_state = gather_clones_for_duct(G_puberty, duct_id, puberty_clones=True)
    pubertal_clones_over_iterations.append(pubertal_state)
    print(f"Puberty state: {len(pubertal_state)} pubertal clones.")

    # Then, for each snapshot from the adulthood simulation.
    # Note: round_graphs[0] is the state after 0 rounds.
    for i in range(n_iterations + 1):
        snapshot = round_graphs[i]
        clones_pubertal = gather_clones_for_duct(snapshot, duct_id, puberty_clones=True)
        pubertal_clones_over_iterations.append(clones_pubertal)
        print(f"Iteration {i}: {len(clones_pubertal)} pubertal clones.")

    # For adult clones (available only in the adulthood snapshots):
    adult_clones_over_iterations = []
    for i in range(n_iterations + 1):
        snapshot = round_graphs[i]
        clones_adult = gather_clones_for_duct(snapshot, duct_id, puberty_clones=False)
        adult_clones_over_iterations.append(clones_adult)
        print(f"Iteration {i}: {len(clones_adult)} adult clones.")

    # --- Build Unified Color Maps ---
    # Pubertal clones color map.
    union_pubertal = []
    for clones in pubertal_clones_over_iterations:
        union_pubertal.extend(clones)
    unique_pubertal = extract_unique_clone_order(union_pubertal)
    pubertal_color_map = {str(cid): cc.glasbey[i] for i, cid in enumerate(unique_pubertal)}

    # Adult clones color map.
    union_adult = []
    for clones in adult_clones_over_iterations:
        union_adult.extend(clones)
    unique_adult = extract_unique_clone_order(union_adult)
    adult_color_map = {str(cid): cc.glasbey[i] for i, cid in enumerate(unique_adult)}

    # --- Create Vertical (Single Duct) Plot for Pubertal Clones ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    # Plot the pubertal state at x = -2 (further separated).
    plot_ductal_clone_line_at_x(pubertal_clones_over_iterations[0], ax1, pubertal_color_map,
                                x=-2, line_width=14, default_color="black")
    # Plot adulthood iterations (from round_graphs) at x = 0, 1, 2, ...
    # (Iteration 1 from round_graphs is plotted at x = 0, iteration 2 at x = 1, etc.)
    for i in range(1, len(pubertal_clones_over_iterations)):
        x_val = i - 1
        plot_ductal_clone_line_at_x(pubertal_clones_over_iterations[i], ax1, pubertal_color_map,
                                    x=x_val, line_width=14, default_color="black")
    # Set x-axis ticks: x = -2 for "pubertal", then 0, 1, … for iterations.
    xticks = [-2] + list(range(0, n_iterations))
    xtick_labels = ["pubertal"] + [str(i) for i in range(n_iterations)]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Normalized Duct Length")
    ax1.set_ylim(1, 0)  # Invert y-axis: 1 at bottom, 0 at top.
    ax1.set_title("Evolution of Pubertal Clones Over Adulthood Iterations (Single Duct)")
    plt.tight_layout()
    pubertal_output = "pubertal_clones_single_duct.png"
    plt.savefig(pubertal_output, dpi=300, bbox_inches="tight")
    print(f"Pubertal clones single duct plot saved to {pubertal_output}")
    plt.close(fig1)

    # --- Create Vertical (Single Duct) Plot for Adult Clones ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for i, clones_iter in enumerate(adult_clones_over_iterations):
        plot_ductal_clone_line_at_x(clones_iter, ax2, adult_color_map,
                                    x=i, line_width=14, default_color="black")
    ax2.set_xticks(range(n_iterations + 1))
    ax2.set_xticklabels([str(i) for i in range(n_iterations + 1)])
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Normalized Duct Length")
    ax2.set_ylim(1, 0)
    ax2.set_title("Evolution of Adult Clones Over Adulthood Iterations (Single Duct)")
    plt.tight_layout()
    adult_output = "adult_clones_single_duct.png"
    plt.savefig(adult_output, dpi=300, bbox_inches="tight")
    print(f"Adult clones single duct plot saved to {adult_output}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
