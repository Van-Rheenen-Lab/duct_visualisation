import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from puberty_population_dynamics import simulate_ductal_tree
from duct_excision_single_hits import compute_single_hit_ratios


from pathlib import Path
file_path = Path(r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\sequencing_data\mutation_matrix_updated_full.txt')

def get_sample_names(file_path):
    with file_path.open() as fh:
        header_line = fh.readline().strip()
    sample_names = [s.strip('"') for s in header_line.split()]
    return sample_names

sample_names = get_sample_names(file_path)
cols = ['Mutation'] + sample_names

df = pd.read_csv(
    file_path,
    sep=r'\s+',
    engine='python',
    quoting=3,
    names=cols,
    skiprows=1
)

# Use the columns of interest
ducts = ['HM6', 'HM8', 'HM11', 'HM15']
data = df[ducts].astype(float)

# Apply VAF threshold
vaf_threshold = 0
mask_nonzero = (data > 0.1).any(axis=1)
data = data.loc[mask_nonzero]

df_clean = data.copy()  # For downstream analysis, use this as the VAF matrix

def compute_real_data_ratios(df, ducts, threshold=0):
    """
    Compute the "single-positive" ratio for each combination of ducts.
    Ratio = (# clones positive in exactly one duct) / (# clones positive in at least one duct)
    Returns:
      - mean_ratios: dict mapping subset_size -> average ratio
      - ratio_points: array of tuples (subset_size, ratio) for each combination
    """
    ratio_points = []
    for r in range(1, len(ducts) + 1):
        for comb in itertools.combinations(ducts, r):
            presence = (df[list(comb)] > threshold)
            count_positive = presence.sum(axis=1)
            total_positive = (count_positive >= 1).sum()
            if total_positive == 0:
                ratio = 0.0
            else:
                single_positive = (count_positive == 1).sum()
                ratio = single_positive / total_positive
            ratio_points.append((r, ratio))
    ratio_points = np.array(ratio_points)

    mean_ratios = {}
    for r in range(1, len(ducts) + 1):
        r_values = ratio_points[ratio_points[:, 0] == r, 1]
        mean_ratios[r] = r_values.mean() if len(r_values) > 0 else np.nan
    return mean_ratios, ratio_points


def compute_real_data_multi_not_all_ratios(df, ducts, threshold=0):
    """
    Compute the "multiple-positive but not all positive" ratio for each combination.
    For a combination of r ducts:
      Ratio = (# clones positive in >=2 but < r ducts) / (# clones positive in at least one duct)
    For r=1, the ratio is defined as 0.

    Returns:
      - mean_ratios: dict mapping subset_size -> average ratio over all combinations
      - ratio_points: array of tuples (subset_size, ratio) for each combination
    """
    ratio_points = []
    for r in range(1, len(ducts) + 1):
        for comb in itertools.combinations(ducts, r):
            presence = (df[list(comb)] > threshold)
            count_positive = presence.sum(axis=1)
            total_positive = (count_positive >= 1).sum()
            if total_positive == 0:
                ratio = 0.0
            else:
                # Count clones that are positive in at least 2 ducts but not in all r ducts.
                if r == 1:
                    ratio = 0.0
                else:
                    multi_not_all = ((count_positive >= 2) & (count_positive < r)).sum()
                    ratio = multi_not_all / total_positive
            ratio_points.append((r, ratio))
    ratio_points = np.array(ratio_points)

    mean_ratios = {}
    for r in range(1, len(ducts) + 1):
        r_values = ratio_points[ratio_points[:, 0] == r, 1]
        mean_ratios[r] = r_values.mean() if len(r_values) > 0 else np.nan
    return mean_ratios, ratio_points


# Compute ratios for real data
real_mean_ratios_single, real_ratio_points_single = compute_real_data_ratios(df_clean, ducts, threshold=0)

# Exclude 0111 pattern from pubertal fraction
# Compute binary pattern for each row
pattern = df_clean.apply(lambda row: ''.join('1' if v > 0 else '0' for v in row), axis=1)
df_clean_no_0111 = df_clean[pattern != '0111']

real_mean_ratios_multi, real_ratio_points_multi = compute_real_data_multi_not_all_ratios(df_clean_no_0111, ducts, threshold=0)


def compute_multi_not_all_hit_ratios(G, max_ducts=None, random_seed=42):
    """
    For simulation data: compute, for subset sizes 1..max_ducts, the ratio:
      Ratio = (# clones present in >=2 but not in all selected ducts) / (# clones present in at least one duct)
    """
    random.seed(random_seed)

    # Identify ducts with clones
    all_ducts = []
    for parent, child in G.edges:
        duct_clones = G[parent][child].get("duct_clones", [])
        if len(duct_clones) > 0:
            all_ducts.append(child)
    unique_ducts = list(set(all_ducts))

    if max_ducts is None:
        max_ducts = len(unique_ducts)

    # Optionally skip the early ducts (as in your original simulation code)
    unique_ducts = unique_ducts[10:]
    random.shuffle(unique_ducts)

    results = []
    for subset_size in range(1, max_ducts + 1):
        selected_ducts = unique_ducts[:subset_size]
        clone_presence = {}
        for duct_node in selected_ducts:
            parents = list(G.predecessors(duct_node))
            for p in parents:
                duct_clones = G[p][duct_node].get("duct_clones", [])
                for c in set(duct_clones):
                    if c not in clone_presence:
                        clone_presence[c] = set()
                    clone_presence[c].add(duct_node)
        if len(clone_presence) == 0:
            ratio = 0.0
        else:
            if subset_size == 1:
                ratio = 0.0
            else:
                multi_not_all = sum(1 for ducts_set in clone_presence.values()
                                    if len(ducts_set) >= 2 and len(ducts_set) < subset_size)
                total_positive = len(clone_presence)
                ratio = multi_not_all / total_positive
        results.append((subset_size, ratio))
    return pd.DataFrame(results, columns=["subset_size", "ratio"])


num_reps = 12
sim_dfs_single = []
sim_dfs_multi = []

# Simulation parameters (as before)
n_clones = 170
max_cells = 3_000_000
bifurcation_prob = 0.01
initial_side_count = n_clones / 2
initial_center_count = n_clones / 2
initial_termination_prob = 0.25
final_termination_prob = 0.55

for i in range(num_reps):
    random.seed(i)
    # Ensure a sufficiently large graph
    while True:
        G_puberty, _ = simulate_ductal_tree(
            max_cells=max_cells,
            bifurcation_prob=bifurcation_prob,
            initial_side_count=initial_side_count,
            initial_center_count=initial_center_count,
            initial_termination_prob=initial_termination_prob,
            final_termination_prob=final_termination_prob,
        )
        if len(G_puberty.nodes) > 80:
            break
        else:
            random.seed(random.randint(0, 1000))
            print("Retrying puberty simulation due to small graph size.")
    # Compute single-hit ratios (existing function) for up to 6 ducts
    df_ratios_single = compute_single_hit_ratios(G_puberty, max_ducts=6)
    sim_dfs_single.append(df_ratios_single)

    # Compute multi-not-all ratios for simulation for up to 6 ducts
    df_ratios_multi = compute_multi_not_all_hit_ratios(G_puberty, max_ducts=6)
    sim_dfs_multi.append(df_ratios_multi)

sim_all_single = pd.concat(sim_dfs_single)
sim_mean_single = sim_all_single.groupby("subset_size")["ratio"].mean()

sim_all_multi = pd.concat(sim_dfs_multi)
sim_mean_multi = sim_all_multi.groupby("subset_size")["ratio"].mean()


# Create the output folder if it doesn't exist
output_folder = "output_plots"
os.makedirs(output_folder, exist_ok=True)

# Only use subset sizes 1 to 4 for plotting and simulation aggregation
max_subset_size = 4

# For simulation, only keep subset sizes 1 to 4
sim_mean_single = sim_mean_single.loc[sim_mean_single.index <= max_subset_size]
sim_mean_multi = sim_mean_multi.loc[sim_mean_multi.index <= max_subset_size]

# For real data, only keep subset sizes 1 to 4
real_subset_sizes = [r for r in sorted(real_mean_ratios_single.keys()) if r <= max_subset_size]
real_values = [real_mean_ratios_single[r] for r in real_subset_sizes]
real_subset_sizes_multi = [r for r in sorted(real_mean_ratios_multi.keys()) if r <= max_subset_size]
real_values_multi = [real_mean_ratios_multi[r] for r in real_subset_sizes_multi]

# Plot 1: Averaged Single-Positive Ratio (Simulation vs Real Data)
plt.figure(figsize=(8, 5))
plt.scatter(sim_mean_single.index, sim_mean_single.values, linestyle='None', color='black',
            alpha=0.6, edgecolor='k', label='Simulation Average Single-Positive Ratio')
plt.scatter(real_subset_sizes, real_values, linestyle='None', color='blue', alpha=0.6,
            edgecolor='k', label='Real Data Average Single-Positive Ratio')
plt.xlabel("Number of Selected Ducts")
plt.ylabel("Ratio")
plt.title("Single-Positive Ratio: Simulation vs. Real Data")
plt.legend()
plt.xticks(np.arange(1, max_subset_size + 1, 1))
plt.savefig(os.path.join(output_folder, "plot1_single_positive_ratio_averaged.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Real Data Single-Positive Ratio (All Combinations)
plt.figure(figsize=(8, 5))
plt.scatter(real_ratio_points_single[:, 0], real_ratio_points_single[:, 1],
            alpha=0.6, color='green', edgecolor='k')
plt.xlabel("Number of Selected Ducts")
plt.ylabel("Single-Positive Ratio")
plt.title("Real Data Single-Positive Ratio (All Combinations)")
plt.xticks(np.arange(1, max_subset_size + 1, 1))
plt.savefig(os.path.join(output_folder, "plot2_single_positive_ratio_all_combinations.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Averaged Multi-Positive (Not All) Ratio (Simulation vs Real Data)
plt.figure(figsize=(8, 5))
plt.scatter(sim_mean_multi.index, sim_mean_multi.values, linestyle='None', color='black',
            alpha=0.6, edgecolor='k', label='Simulation Average Multi-Positive (Not All) Ratio')
plt.scatter(real_subset_sizes_multi, real_values_multi, linestyle='None', color='blue', alpha=0.6,
            edgecolor='k', label='Real Data Average Multi-Positive (Not All) Ratio')
plt.xlabel("Number of Selected Ducts")
plt.ylabel("Ratio")
plt.title("Multi-Positive (But Not All) Ratio: Simulation vs. Real Data (Averaged)")
plt.legend()
plt.xticks(np.arange(1, max_subset_size + 1, 1))
plt.savefig(os.path.join(output_folder, "plot3_multi_positive_ratio_averaged.png"), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Real Data Multi-Positive (Not All) Ratio (All Combinations)
plt.figure(figsize=(6, 4))
plt.scatter(real_ratio_points_multi[:, 0], real_ratio_points_multi[:, 1],
            alpha=0.6, color='purple', edgecolor='k')
plt.xlabel("Number of Selected Ducts")
plt.ylabel("Pubertal fraction")
plt.title("Puberty-assigned fraction for increasing number of samples")
plt.xticks(np.arange(1, max_subset_size + 1, 1))
plt.savefig(os.path.join(output_folder, "plot4_multi_positive_ratio_all_combinations.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_folder, "plot4_multi_positive_ratio_all_combinations.svg"))
plt.close()

# Apply cyclic right shift to columns for barcode pattern assignment
barcode_columns = ducts[-1:] + ducts[:-1]
barcode_data = df_clean[barcode_columns]

# Assign patterns as in the analysis script
pattern = barcode_data.apply(lambda row: ''.join('1' if v > 0 else '0' for v in row), axis=1)
excluded = {'0111'}
sel = pattern.str.count('1').between(2, 3) & ~pattern.isin(excluded)
pubertal_fraction_4ducts = sel.mean()
print(f"Real data pubertal fraction at 4 ducts (excluding 0111, analysis style, cyclic shift): {pubertal_fraction_4ducts}")
print(f"Total pubertal VAFs: {sel.sum()} out of {len(sel)}")
# Save to file for use in other scripts
with open('pubertal_fraction_4ducts.txt', 'w') as f:
    f.write(str(pubertal_fraction_4ducts))

print("Rows with all zeros in 4 ducts after filtering:", (df_clean == 0).all(axis=1).sum())
