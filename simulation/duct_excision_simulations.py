import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Import your custom simulation and plotting
from puberty import simulate_ductal_tree
from adulthood import simulate_adulthood
from plotting_simulated_ducts import plot_selected_ducts

def gather_clone_fractions_for_selected_ducts(G, selected_ducts, n_clones=100):
    """
    Build a DataFrame with shape (n_clones x len(selected_ducts)),
    whose (row=clone_id, column=duct_node) entry is the fraction of that clone
    among all cells in the edge (parent -> duct_node).
    """
    clone_ids = list(range(n_clones))
    df = pd.DataFrame(index=clone_ids, columns=selected_ducts, data=0.0)

    for child_node in selected_ducts:
        if child_node not in G.nodes:
            print(f"Node {child_node} not in graph, skipping.")
            continue

        parents = list(G.predecessors(child_node))
        if not parents:
            # No parent means this node might be the root or disconnected
            continue

        parent = parents[0]
        duct_clones = G[parent][child_node].get("duct_clones", [])
        total_cells = len(duct_clones)
        if total_cells == 0:
            continue

        # Count each clone
        clone_counts = {}
        for c in duct_clones:
            clone_counts[c] = clone_counts.get(c, 0) + 1

        # Convert to fraction of total
        for c, count in clone_counts.items():
            if c < n_clones:
                df.loc[c, child_node] = count / total_cells

    return df

def main():
    random.seed(42)

    n_clones = 400

    G_puberty, _ = simulate_ductal_tree(
        max_cells=300_000,
        bifurcation_prob=0.01,
        initial_side_count=n_clones/2,
        initial_center_count=n_clones/2,
        initial_termination_prob=0.2
    )
    G_puberty_copy = G_puberty.copy()
    G_adulthood, _ = simulate_adulthood(G_puberty_copy, rounds=33)

    # select duct 180 and all its predecessors, and its predecessors' predecessors etc
    selected_ducts = [180]
    while True:
        parents = list(G_puberty.predecessors(selected_ducts[0]))
        if not parents:
            break
        selected_ducts = parents + selected_ducts



    # ---------------------------
    # 3) Plot the ductal trees
    # ---------------------------
    plot_selected_ducts(G_puberty, selected_ducts, vert_gap=2)
    plt.title("Annotated Ducts")

    # ------------------------------------------------------------------------
    # 4) Build DataFrames for clone fractions in selected ducts (Puberty vs Adult)
    # ------------------------------------------------------------------------
    df_puberty = gather_clone_fractions_for_selected_ducts(
        G_puberty, selected_ducts, n_clones=n_clones
    )
    df_adulthood = gather_clone_fractions_for_selected_ducts(
        G_adulthood, selected_ducts, n_clones=n_clones
    )


    vaf_threshold = 0
    vaf_columns = selected_ducts  # or selected_ducts[:6] if you only want 6 bits

    # Generate an ordered pattern list, from 1-positive-bit to 6-positive-bits (if you indeed have 6 columns).
    # Adjust as needed to match the actual number of columns.
    num_bits = len(vaf_columns)
    pattern_order = []
    for r in range(1, num_bits + 1):
        for combo in itertools.combinations(range(num_bits), r):
            pattern = ['0'] * num_bits
            for bit_index in combo:
                pattern[bit_index] = '1'
            pattern_order.append("".join(pattern))
    # Pattern -> cluster index
    pattern_to_cluster = {p: i for i, p in enumerate(pattern_order)}

    # Helper function to assign a pattern to each row
    def assign_pattern(row, threshold=vaf_threshold, columns=vaf_columns):
        bits = []
        for c in columns:
            bits.append('1' if row[c] > threshold else '0')
        return "".join(bits)

    # Assign pattern & cluster
    df_puberty['Pattern'] = df_puberty.apply(assign_pattern, axis=1)
    df_puberty['Cluster'] = df_puberty['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))

    # Sort puberty clones by cluster
    df_puberty.sort_values('Cluster', inplace=True)

    # Reorder the adulthood dataframe by the same clone index order
    # to align the same clones in the same rows.
    df_adulthood = df_adulthood.loc[df_puberty.index]

    # -------------------------------------------------------------
    # 6) Plot heatmaps in the same clone order
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_puberty[selected_ducts], vmin=0.0, vmax=0.2)
    plt.xlabel("Duct ID")
    plt.ylabel("Clones")
    plt.yticks([])
    plt.title("Simulated Sequencing Post-Puberty")

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_adulthood[selected_ducts], vmin=0.0, vmax=0.2)
    plt.xlabel("Duct ID")
    plt.ylabel("Clones")
    plt.yticks([])
    plt.title("Simulated Sequencing Post-Adulthood")
    plt.show()

if __name__ == "__main__":
    main()
