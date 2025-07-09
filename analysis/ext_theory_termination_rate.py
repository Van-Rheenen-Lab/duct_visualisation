import numpy as np
from analysis.utils.cumulative_area import compute_branch_levels
import networkx as nx


def termination_ratio_by_level(G_dir, root_node):
    """
    Edge-centred termination statistics.

    Parameters
    ----------
    G_dir : networkx.DiGraph
        Directed graph of the duct tree (successor = downstream).
    root_node : hashable
        Node taken as level-0.

    Returns
    -------
    ratio : np.ndarray, shape (L+1,)
        Termination ratio R_k for branch levels k = 0 … L
        (NaN where no edge ends at that level).
    term_counts : np.ndarray, shape (L+1,)
        Z_k – number of terminating edges per level.
    edge_counts : np.ndarray, shape (L+1,)
        N_k – total number of edges whose **child** node lies at that level.

    Notes
    -----
    Let ℓ(v) be the branch level of node v, computed with
    ``compute_branch_levels(G_dir, root_node)``.

    For every edge e = (u → v):

        • level(e) := ℓ(v)                            (child’s level)
        • termination indicator T(e) := 1 if v has out-degree 0, else 0

    For level k
        N_k = Σ_{e} [ℓ(v)=k]       (edge count)
        Z_k = Σ_{e} T(e)·[ℓ(v)=k] (terminating edges)

        **Termination ratio** R_k = Z_k / N_k
    """
    # 1. branch levels of nodes
    levels = compute_branch_levels(G_dir, root_node)
    if not levels:                       # empty graph
        return np.array([]), np.array([]), np.array([])

    max_lvl = max(levels.values())
    edge_counts = np.zeros(max_lvl + 1, dtype=np.int64)   # N_k
    term_counts = np.zeros_like(edge_counts)              # Z_k

    # 2. iterate over every directed edge (u → v)
    for u, v in G_dir.edges():
        lvl = levels[v]                 # child’s branch level
        edge_counts[lvl] += 1
        if G_dir.out_degree(v) == 0:    # v is a leaf
            term_counts[lvl] += 1

    # 3. ratio R_k  (NaN where N_k = 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(edge_counts > 0,
                         term_counts / edge_counts,
                         np.nan)
    return ratio, term_counts, edge_counts


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from analysis.utils.loading_saving import load_duct_systems, select_biggest_duct_system, create_directed_duct_graph, find_root

    # set plotting font size
    plt.rcParams.update({'font.size': 18,
                         'font.family': 'Arial',
                         'axes.titlesize': 16,
                         'axes.labelsize': 16,
                         'xtick.labelsize': 16,
                         'ytick.labelsize': 16})


    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_systems = load_duct_systems(json_path)
    duct_system = select_biggest_duct_system(duct_systems)
    G = create_directed_duct_graph(duct_system)
    root = find_root(G)

    ratio, term_counts, edge_counts = termination_ratio_by_level(G, root)
    print("Termination Ratios:", ratio)
    print("Termination Counts:", term_counts)
    print("Edge Counts:", edge_counts)

    # Plotting the termination ratio by level
    levels = np.arange(len(ratio))
    plt.figure(figsize=(10, 6))
    plt.scatter(levels, ratio, color='blue', label='Termination Ratio', s=50)
    plt.xlabel('Time (branch Level)')
    plt.ylabel('Termination Ratio')

    # save as SVG
    plt.savefig("termination_ratio_cft.svg", format="svg", bbox_inches="tight")

    from simulation.puberty_deposit_elimination import simulate_ductal_tree
    import random

    random.seed(41)

    n_clones = 170
    bifurcation_prob = 0.01
    initial_termination_prob = 0.25
    final_termination_prob = 0.55
    max_cells = 3_000_000

    # -- Simulate Puberty --
    G_sim, progress_data = simulate_ductal_tree(
        max_cells=max_cells,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=n_clones / 2,
        initial_center_count=n_clones / 2,
        initial_termination_prob=initial_termination_prob,
        final_termination_prob=final_termination_prob
    )

    root_sim = find_root(G_sim)
    ratio_sim, term_counts_sim, edge_counts_sim = termination_ratio_by_level(G_sim, root_sim)
    levels = np.arange(len(ratio_sim))
    plt.figure(figsize=(10, 6))
    plt.scatter(levels, ratio_sim, color='red', label='Termination Ratio (Simulated)', s=50)
    plt.xlabel('Time (branch Level)')
    plt.ylabel('Termination Ratio')
    plt.savefig("termination_ratio_simulated.svg", format="svg", bbox_inches="tight")
    plt.show()
