import random
import pandas as pd
import matplotlib.pyplot as plt
from puberty import simulate_ductal_tree

import random
import pandas as pd


def compute_single_hit_ratios(G, max_ducts=None, random_seed=42):
    """
    Computes, for an increasing subset of ducts (1..max_ducts):
      ratio = (# clones found in exactly one of those ducts) / (# clones found in at least one duct)

    *Important*: We treat each clone as "present" or "absent" in a duct,
                 ignoring how many times it appears within that duct.
    """
    random.seed(random_seed)

    # -------------------------------------------------
    # Identify all ducts that have duct_clones
    # -------------------------------------------------
    all_ducts = []
    for parent, child in G.edges:
        duct_clones = G[parent][child].get("duct_clones", [])
        if len(duct_clones) > 0:
            all_ducts.append(child)
    unique_ducts = list(set(all_ducts))

    if max_ducts is None:
        max_ducts = len(unique_ducts)

    # -------------------------------------------------
    # Shuffle and slice subsets
    # -------------------------------------------------
    # Truncate the first n off, to avoid early termination results
    unique_ducts = unique_ducts[10:]
    random.shuffle(unique_ducts)

    results = []
    for subset_size in range(1, max_ducts + 1):
        selected_ducts = unique_ducts[:subset_size]

        # Track which ducts each clone appears in:
        # clone_presence[c] = set of duct IDs in which clone c is found
        clone_presence = {}

        for duct_node in selected_ducts:
            parents = list(G.predecessors(duct_node))
            for p in parents:
                duct_clones = G[p][duct_node].get("duct_clones", [])
                # Convert to a set so we only mark presence once per duct,
                # ignoring if a clone appears multiple times in the same duct.
                for c in set(duct_clones):
                    if c not in clone_presence:
                        clone_presence[c] = set()
                    clone_presence[c].add(duct_node)

        if len(clone_presence) == 0:
            ratio = 0.0
        else:
            single_positive = sum(1 for ducts_set in clone_presence.values() if len(ducts_set) == 1)
            total_positive = len(clone_presence)
            ratio = single_positive / total_positive

        results.append((subset_size, ratio))

    return pd.DataFrame(results, columns=["subset_size", "ratio"])


def main():
    # Number of repeated simulations
    num_reps = 12

    # Settings for puberty simulation
    n_clones = 100
    max_cells = 200_000
    bifurcation_prob = 0.01
    initial_side_count = n_clones / 2
    initial_center_count = n_clones / 2
    initial_termination_prob = 0.1

    # Collect the ratio DataFrames for each replicate
    list_of_dfs = []

    for i in range(num_reps):
        # Each replicate: simulate puberty once
        random.seed(i)

        while True:
            G_puberty, _ = simulate_ductal_tree(
                max_cells=max_cells,
                bifurcation_prob=bifurcation_prob,
                initial_side_count=initial_side_count,
                initial_center_count=initial_center_count,
                initial_termination_prob=initial_termination_prob
            )
            if len(G_puberty.nodes) > 80:
                break
            else:
                del G_puberty
                random.seed(random.randint(0, 1000))
                print("Retrying puberty simulation due to small graph size.")

        # Compute the single-hit ratio vs subset size
        # Provide a different random_seed each time if you want truly different shuffles:
        df_ratios = compute_single_hit_ratios(G_puberty, max_ducts=30)
        list_of_dfs.append(df_ratios)

    plt.figure(figsize=(10, 6))

    # Plot each replicate's data as scatter
    for idx, df in enumerate(list_of_dfs):
        plt.scatter(
            df["subset_size"],
            df["ratio"],
            alpha=0.2,
            label=f"Replicate {idx+1}"
        )

    # add 1 point how our real data looks like: 67.54% at 4 ducts
    plt.scatter(4, 0.6754, color='red', label='Data Patient 2')

    # Compute the average ratio across replicates
    df_concat = pd.concat(list_of_dfs)
    df_mean = df_concat.groupby("subset_size")["ratio"].mean()

    # Plot the mean ratio line
    plt.plot(
        df_mean.index,
        df_mean.values,
        color="black",
        linewidth=2,
        label="Mean ratio"
    )

    plt.xlabel("Number of Randomly Selected Ducts (No Early Ducts Selected)")
    plt.ylabel("Ratio (# progeny in exactly one duct / # progeny in at least one duct)")
    plt.title("Single-Hit Ratio vs. Number of Selected Ducts")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
