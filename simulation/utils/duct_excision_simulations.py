import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from simulation.puberty import simulate_ductal_tree
from simulation.adulthood import simulate_adulthood
from simulation.utils.plotting_simulated_ducts import plot_selected_ducts


def gather_clone_fractions_for_selected_ducts(G, selected_ducts, puberty_clones=True, restrict_to_selected=False):
    """
    Build a DataFrame (rows = clone_id, columns = duct nodes) where each entry is
    the fraction of that clone among the cells in the edge (parent -> duct_node).

    Parameters:
      G                 : Graph representing the ductal tree.
      selected_ducts    : List of duct node IDs for which to gather fractions.
      puberty_clones    : If True, use the 'duct_clones' attribute (pubertal IDs);
                          otherwise, use 'adult_clones'.
      restrict_to_selected : If True, only use clone IDs from edges whose child node is
                             in selected_ducts.
    """
    clone_ids = set()
    for (u, v) in G.edges():
        if restrict_to_selected and v not in selected_ducts:
            continue  # skip edges not ending in one of the selected ducts
        if puberty_clones:
            duct_clones = G[u][v].get("duct_clones", [])
        else:
            duct_clones = G[u][v].get("adult_clones", [])
        clone_ids.update(duct_clones)

    def sort_key(x):
        # If it's an integer, just return (0, integer_value)
        if isinstance(x, int):
            return (0, x)
        # If it's a string like A_2, parse out the numeric portion to sort properly
        if isinstance(x, str) and x.startswith("A_"):
            suffix = x.split("_")[1]
            return (1, int(suffix))
        # fallback
        return (2, str(x))

    index = sorted(clone_ids, key=sort_key)
    df = pd.DataFrame(index=index, columns=selected_ducts, data=0.0)

    # Loop over each selected duct and compute clone fractions on the incoming edge.
    for child_node in selected_ducts:
        if child_node not in G.nodes:
            print(f"Node {child_node} not in graph, skipping.")
            continue

        parents = list(G.predecessors(child_node))
        if not parents:
            continue  # no parent => skip

        parent = parents[0]
        if puberty_clones:
            duct_clones = G[parent][child_node].get("duct_clones", [])
        else:
            duct_clones = G[parent][child_node].get("adult_clones", [])

        total_cells = len(duct_clones)
        print(len(duct_clones))
        if total_cells == 0:
            continue

        # Count each clone
        clone_counts = {}
        for c in duct_clones:
            clone_counts[c] = clone_counts.get(c, 0) + 1

        # Convert counts to fractions and store in the DataFrame
        for c, count in clone_counts.items():
            df.loc[c, child_node] = count / total_cells

    return df


def main():
    random.seed(42)

    n_clones = 170

    # Simulate the puberty (ductal tree) phase.
    G_puberty, _ = simulate_ductal_tree(
        max_cells=5_000_000,
        bifurcation_prob=0.01,
        initial_side_count=n_clones / 2,
        initial_center_count=n_clones / 2,
        initial_termination_prob=0.3
    )
    G_puberty_copy = G_puberty.copy()

    # Simulate adulthood on a copy of the puberty graph.
    G_adulthood, _ = simulate_adulthood(G_puberty_copy, rounds=33)


    last_bp = list(G_puberty.nodes)[-1]
    selected_ducts = [last_bp]

    # Don't do the first few (5), because we don't analyse that in our own experimental data.
    # Also take one in 3 to make sure we have no local overlaps
    while True:
        parents = list(G_puberty.predecessors(selected_ducts[0]))
        if not parents:
            break
        selected_ducts = parents + selected_ducts

    # Remove the root (node 0) if desired
    selected_ducts = selected_ducts[5:]
    # Select every 3rd duct
    selected_ducts = selected_ducts[::3]

    plot_selected_ducts(G_puberty, selected_ducts, vert_gap=2)
    plt.title("Annotated Ducts")


    # (1) For the original puberty simulation.
    df_puberty = gather_clone_fractions_for_selected_ducts(
        G_puberty, selected_ducts, puberty_clones=True
    )
    # (2) For post-adulthood: pubertal clone IDs (duct_clones from G_adulthood)
    df_adulthood_pubertal = gather_clone_fractions_for_selected_ducts(
        G_adulthood, selected_ducts, puberty_clones=True
    )
    # (3) For post-adulthood: adult clone IDs (adult_clones from G_adulthood)
    df_adulthood_adult = gather_clone_fractions_for_selected_ducts(
        G_adulthood, selected_ducts, puberty_clones=False, restrict_to_selected=True
    )

    vaf_threshold = 0
    vaf_columns = selected_ducts  # columns used to compute pattern

    num_bits = len(vaf_columns)
    pattern_order = []
    for r in range(1, num_bits + 1):
        for combo in itertools.combinations(range(num_bits), r):
            pattern = ['0'] * num_bits
            for bit_index in combo:
                pattern[bit_index] = '1'
            pattern_order.append("".join(pattern))
    pattern_to_cluster = {p: i for i, p in enumerate(pattern_order)}

    def assign_pattern(row, threshold=vaf_threshold, columns=vaf_columns):
        bits = []
        for c in columns:
            bits.append('1' if row[c] > threshold else '0')
        return "".join(bits)

    df_puberty['Pattern'] = df_puberty.apply(assign_pattern, axis=1)
    df_puberty['Cluster'] = df_puberty['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))

    # Sort the puberty DataFrame by cluster (for consistent ordering)
    df_puberty.sort_values('Cluster', inplace=True)

    # Reindex the post-adulthood pubertal clones to follow the same ordering.
    df_adulthood_pubertal = df_adulthood_pubertal.reindex(df_puberty.index, fill_value=0.0)

    df_adulthood_adult['Pattern'] = df_adulthood_adult.apply(assign_pattern, axis=1)
    df_adulthood_adult['Cluster'] = df_adulthood_adult['Pattern'].apply(lambda x: pattern_to_cluster.get(x, -1))

    df_adulthood_adult.sort_values('Cluster', inplace=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_puberty[selected_ducts], vmin=0.0, vmax=0.2)
    plt.xlabel("Duct ID")
    plt.ylabel("Clones")
    plt.yticks([])
    plt.title("Simulated Sequencing Post-Puberty")

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_adulthood_pubertal[selected_ducts], vmin=0.0, vmax=0.2)
    plt.xlabel("Duct ID")
    plt.ylabel("Clones (Pubertal IDs in Adulthood)")
    plt.yticks([])
    plt.title("Simulated Sequencing Post-Adulthood (Pubertal Clones)")

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_adulthood_adult[selected_ducts], vmin=0.0, vmax=0.2)
    plt.xlabel("Duct ID")
    plt.ylabel("Clones (Adult IDs)")
    plt.yticks([])
    plt.title("Simulated Sequencing Post-Adulthood (Adult Clones)")

    plt.show()


if __name__ == "__main__":
    main()
