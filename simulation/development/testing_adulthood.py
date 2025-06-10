import matplotlib.pyplot as plt
import numpy as np
import random

from simulation.puberty import simulate_ductal_tree
from simulation.adulthood import simulate_adulthood
from simulation.utils.plotting_simulated_ducts import plotting_ducts
from scipy.special import erfc
from scipy.optimize import curve_fit

def plot_stack_counts_with_fixed_ids(iterations, dist_series, all_ids, title, color_map=None):
    """
    Plots a stackplot of distribution data (dist_series) across the given
    fixed list of IDs (all_ids), ensuring consistent ordering.

    Parameters
    ----------
    iterations : list of int
        X-axis values (e.g., iteration indices).
    dist_series : list of dict
        Each entry is a dict of the form {clone_id: count, ...}
        representing the distribution at that iteration.
    all_ids : list
        The sorted list of IDs (strings or ints) to plot in a fixed order.
    title : str
        Plot title.
    color_map : list or None
        Optional list of colors to use, must have length >= len(all_ids).
        If None, we will generate a default colormap.
    """
    # Build matrix of shape (#all_ids x #iterations)
    data_matrix = np.zeros((len(all_ids), len(iterations)), dtype=int)
    for col_i, distribution_dict in enumerate(dist_series):
        for row_i, the_id in enumerate(all_ids):
            data_matrix[row_i, col_i] = distribution_dict.get(the_id, 0)

    # If no color_map provided, build one from a default colormap
    if color_map is None:
        cmap = plt.get_cmap("tab20")  # or "hsv", "rainbow", etc.
        color_map = [cmap(i % 20) for i in range(len(all_ids))]

    plt.figure(figsize=(8, 5))
    plt.stackplot(iterations, data_matrix, labels=all_ids, colors=color_map[:len(all_ids)])
    plt.xlabel("Iteration")

    plt.title(title)
    # Legend can be large if many IDs
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

random.seed(42)
n_clones = 170

# 1) Run puberty simulation
G, progress_data_puberty = simulate_ductal_tree(
    max_cells=5_000_000,
    bifurcation_prob=0.01,
    initial_side_count=n_clones/2,
    initial_center_count=n_clones/2,
    initial_termination_prob=0.2
)

# make a color hex dict with tab10 colors, but in hex in dict
color_map = {"42": "#1f77b4", "84": "#ff7f0e", "12": "#2ca02c", "24": "#d62728"}


plotting_ducts(G, vert_gap=5, color_map=color_map)
plt.title("Ductal Tree at after Puberty, before Adulthood")


# 2) Run adulthood simulation
remodel_length = 20 # note that this is in stem cells. So real cells is *10
remodel_prob = 0.1
G, progress_data_adult = simulate_adulthood(G, rounds=33, remodel_length=remodel_length, remodel_prop=remodel_prob)
plotting_ducts(G, vert_gap=5, color_map=color_map)
plt.title("Ductal Tree at after Adulthood")


# Extract puberty iteration data
puberty_iters = progress_data_puberty["iteration"]
puberty_dists = progress_data_puberty["clone_counts_over_time"]  # list of dicts (pubertal_id->count)
num_active_tebs = progress_data_puberty["num_active_tebs"]

# Extract adulthood iteration data
adult_iters = progress_data_adult["iteration"]
adult_pub_dists = progress_data_adult["pubertal_id_counts"]  # list of dicts (pubertal_id->count)
adult_adult_dists = progress_data_adult["adult_id_counts"]  # list of dicts (adult_id->count)

# ----------------------------------------------------------------------------
# 3) Collect the *unified* set of pubertal IDs across both phases
# ----------------------------------------------------------------------------
all_pub_ids = set()
for dist_dict in puberty_dists:
    all_pub_ids.update(dist_dict.keys())
for dist_dict in adult_pub_dists:
    all_pub_ids.update(dist_dict.keys())

# Convert to a sorted list
all_pub_ids = sorted(all_pub_ids)

# ----------------------------------------------------------------------------
# 4) Plot puberty stack (pubertal IDs) *with fixed ordering*
# ----------------------------------------------------------------------------
plot_stack_counts_with_fixed_ids(
    iterations=puberty_iters,
    dist_series=puberty_dists,
    all_ids=all_pub_ids,
    title="Puberty Phase (Pubertal IDs)"
)
plt.ylabel("Cell Count")
# ----------------------------------------------------------------------------
# 5) Plot adulthood stack of pubertal IDs (same ordering, same color map)
# ----------------------------------------------------------------------------
# If you want the exact same color assignments for each pubertal ID across
# puberty and adulthood, you can pre-build a color map once and pass it again.

# Build a consistent color map. For example:
cmap = plt.get_cmap("tab20")
pubertal_color_map = [cmap(i % 20) for i in range(len(all_pub_ids))]

plot_stack_counts_with_fixed_ids(
    iterations=adult_iters,
    dist_series=adult_pub_dists,
    all_ids=all_pub_ids,
    title="Adult Phase (Pubertal IDs)",
    color_map=pubertal_color_map
)
plt.ylabel("Stem Cell Count")
# ----------------------------------------------------------------------------
# 6) For adult IDs, we can do a separate plot (they are distinct from pubertal IDs)
# ----------------------------------------------------------------------------
# gather all adult IDs
all_adult_ids = set()
for dist_dict in adult_adult_dists:
    all_adult_ids.update(dist_dict.keys())
all_adult_ids = sorted(all_adult_ids)

# plot number of unique adult IDs per iteration

plt.figure(figsize=(8, 5))
plt.plot(adult_iters, [len(d) for d in adult_adult_dists], label="Unique Adult IDs", color="red")
plt.xlabel("Iteration")
plt.ylabel("Number of Unique Adult IDs")
plt.title("Unique Adult IDs Over Time")

# plot unique pubertal IDs
plt.figure(figsize=(8, 5))
plt.plot(adult_iters, [len(d) for d in adult_pub_dists], label="Unique Pubertal IDs", color="blue")
plt.xlabel("Iteration")
plt.ylabel("Number of Unique Pubertal IDs in total duct system")
plt.title("Unique Pubertal IDs Over Time in total duct")

plt.figure(figsize=(8, 5))
plt.plot(puberty_iters, num_active_tebs, label="Active TEBs", color="green")
plt.xlabel("Iteration")
plt.ylabel("Number of TEBs")
plt.title("Active TEBs Over Time")
plt.legend()
plt.tight_layout()

# calculate the average number of stem cells per unique adult ID
avg_stem_cells = []
for dist_dict in adult_adult_dists:
    stem_count = 0
    for adult_id, count in dist_dict.items():

        stem_count += count
    avg_stem_cells.append(stem_count / len(dist_dict))

plt.figure(figsize=(8, 5))
plt.plot(adult_iters, avg_stem_cells, label="Average Stem Cells per Adult ID", color="blue")
plt.xlabel("Iteration")
plt.ylabel("Average Stem Cells")
plt.title("Average Stem Cells per Unique Adult ID Over Time")


# plot distribution of number of stem cells per unique adult id
plt.figure(figsize=(8, 5))
distribution = adult_adult_dists[-1].values()
average = sum(distribution) / len(distribution)
print(f"Average of stem cells per unique adult id: {average}")
bins = range(min(distribution), max(distribution) + 2)
plt.hist(distribution, bins=bins, align='left', rwidth=0.8)
# plot red line for average
plt.axvline(average, color='red', linestyle='dashed', linewidth=2)
# make legend with average
plt.legend([f"Average: {average:.2f}"])

plt.xticks(range(min(distribution), max(distribution) + 1))
plt.xlabel("Number of stem cells in adult clone")
plt.ylabel("Frequency")
plt.title("Stem Cells per Surviving Adult Clone after Adulthood")


unique_clones_per_duct = progress_data_adult["unique_clones_per_duct"]
plt.figure(figsize=(8, 5))
average_clones_per_duct = []
for d in unique_clones_per_duct:
    average_clones_per_duct.append(sum(d.values()) / len(d))
plt.plot(adult_iters, average_clones_per_duct, label="Average Unique Clones IDs per Duct", color="purple")
plt.xlabel("Iteration")
plt.ylabel("Average Clones per Duct")
plt.title("Average Unique Pubertal IDs per Duct Over Time")

# Your clone counts (scaled by 10)
clone_counts = np.array(list(adult_adult_dists[-1].values()))
clone_counts *= 10

# Define the lognormal survival model (theoretical cumulative probability)
def lognormal_survival(x, mu, sigma):
    return 0.5 * erfc((np.log(x) - mu) / (np.sqrt(2) * sigma))

sizes_fit = np.arange(1, int(remodel_length*8)+1)

# Compute the empirical cumulative probability over the fitting range
empirical_cum_prob_fit = np.array([np.sum(clone_counts >= s) / len(clone_counts) for s in sizes_fit])

# Provide an initial guess for mu and sigma based on the truncated data
initial_mu = np.mean(np.log(clone_counts[clone_counts <= remodel_length*8]))
initial_sigma = np.std(np.log(clone_counts[clone_counts <= remodel_length*8]))

# Fit the lognormal model using curve_fit
popt, pcov = curve_fit(
    lognormal_survival, sizes_fit, empirical_cum_prob_fit,
    p0=[initial_mu, initial_sigma]
)
mu_fit, sigma_fit = popt


max_clone_size = clone_counts.max()
sizes_full = np.arange(1, max_clone_size + 1)
empirical_cum_prob_full = np.array([np.sum(clone_counts >= s) / len(clone_counts) for s in sizes_full])

# Now compute the theoretical curve using the fitted parameters over the fitting range
theoretical_cum_prob_fit = lognormal_survival(sizes_full, mu_fit, sigma_fit)

# Plot everything
plt.figure(figsize=(8, 5))
# Empirical data (full range)
plt.loglog(sizes_full, empirical_cum_prob_full, marker='o', ls='None', label="Simulated Data")
# Fitted theoretical curve over the fitting range (1 to 50)
plt.loglog(sizes_full, theoretical_cum_prob_fit, marker=None, ls='-', linewidth=3.0, label="Lognormal Fit")

# vline at 100
plt.axvline(x=remodel_length*10, color='red', linestyle='--', label="Expected departure from lognormal")

plt.xlabel("Clone Size")
plt.ylabel("Cumulative Probability")
plt.title("Log-Log Plot of Cumulative Clone Size Distribution")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Optionally, print the fitted parameters for inspection
print(f"Fitted mu: {mu_fit}, Fitted sigma: {sigma_fit}")
