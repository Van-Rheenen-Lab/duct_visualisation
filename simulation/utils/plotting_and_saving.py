import os
import csv
import random
import matplotlib.pyplot as plt
import numpy as np

from puberty import simulate_ductal_tree
from adulthood import simulate_adulthood
from plotting_simulated_ducts import plotting_ducts


def plot_stack_counts_with_fixed_ids(iterations, dist_series, all_ids, title, output_filename,
                                     color_map=None, dpi=300):
    """
    Plots a stackplot of distribution data (dist_series) across the given
    fixed list of IDs (all_ids), ensuring consistent ordering, then saves as PNG.

    Parameters
    ----------
    iterations : list of int
        X-axis values (e.g., iteration indices).
    dist_series : list of dict
        Each entry is a dict of {clone_id: count, ...} for the iteration.
    all_ids : list
        The sorted list of IDs (strings or ints) to plot in a fixed order.
    title : str
        Plot title.
    output_filename : str
        Filename to save the figure (e.g., "puberty_phase.png").
    color_map : list or None
        Optional list of colors to use. Must have length >= len(all_ids).
        If None, a default colormap is generated.
    dpi : int
        Dots per inch for figure saving.
    """
    data_matrix = np.zeros((len(all_ids), len(iterations)), dtype=int)
    for col_i, distribution_dict in enumerate(dist_series):
        for row_i, the_id in enumerate(all_ids):
            data_matrix[row_i, col_i] = distribution_dict.get(the_id, 0)

    # If no color_map provided, build one from a default colormap
    if color_map is None:
        cmap = plt.get_cmap("tab20")
        color_map = [cmap(i % 20) for i in range(len(all_ids))]

    plt.figure(figsize=(8, 5))
    plt.stackplot(iterations, data_matrix, labels=all_ids, colors=color_map[:len(all_ids)])
    plt.xlabel("Time (days)")
    plt.ylabel("Cell Count")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(output_filename, dpi=dpi)
    plt.close()  # Close to avoid overlapping figures in subsequent plots


def export_distribution_csv(iterations, dist_series, csv_filename):
    """
    Exports the distribution data (a list of dicts at each iteration) into a CSV.

    The CSV will have the following columns:
    - iteration
    - clone_id
    - count

    Each row corresponds to one clone in one iteration.

    Parameters
    ----------
    iterations : list of int
    dist_series : list of dict
    csv_filename : str
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "clone_id", "count"])
        for iter_idx, dist_dict in zip(iterations, dist_series):
            for cid, count in dist_dict.items():
                writer.writerow([iter_idx, cid, count])


def main():
    seed = 42
    random.seed(seed)
    n_clones = 170
    output_dir = "simulation_outputs2"
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "csv_data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    bifurcation_prob = 0.01
    initial_termination_prob = 0.25

    # Run puberty simulation
    G, progress_data_puberty = simulate_ductal_tree(
        max_cells=6_000_000,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=n_clones / 2,
        initial_center_count=n_clones / 2,
        initial_termination_prob=initial_termination_prob
    )

    # Fixed custom color map for certain IDs
    color_map = {"84": "#ff7f0e", "58": "#2ca02c", "21": "#b02797"}

    # Plot the ductal tree after Puberty
    plt.figure(figsize=(20, 12))
    plotting_ducts(G, vert_gap=5, color_map=color_map)
    plt.title("Ductal Tree after Puberty, before Adulthood")
    plt.savefig(os.path.join(figures_dir, "ductal_tree_after_puberty.png"), dpi=600)
    plt.close()

    G_copy = G.copy()

    # Run adulthood simulation
    G, progress_data_adult = simulate_adulthood(G, rounds=33)

    plt.figure(figsize=(20, 12))
    plotting_ducts(G, vert_gap=5, color_map=color_map)
    plt.title("Ductal Tree after Adulthood")
    plt.savefig(os.path.join(figures_dir, "ductal_tree_after_adulthood.png"), dpi=600)
    plt.close()

    # Extract puberty iteration data
    puberty_iters = progress_data_puberty["iteration"]
    # Rescale time axis for puberty to 4 weeks, So total time is 4*7 days for puberty. 33*4.5 days for adulthood,
    puberty_days = 4 * 7
    puberty_iters = [(i/max(puberty_iters)) * puberty_days for i in puberty_iters]

    puberty_dists = progress_data_puberty["clone_counts_over_time"]  # list of dicts
    num_active_tebs = progress_data_puberty["num_active_tebs"]

    # Extract adulthood iteration data
    adult_iters = progress_data_adult["iteration"]
    adult_days = 33 * 4.5
    adult_iters = [i * 4.5 for i in adult_iters]
    adult_pub_dists = progress_data_adult["pubertal_id_counts"]  # list of dicts
    adult_adult_dists = progress_data_adult["adult_id_counts"]   # list of dicts

    export_distribution_csv(puberty_iters, puberty_dists,
                            os.path.join(data_dir, "puberty_pubertal_ids_distribution.csv"))

    export_distribution_csv(adult_iters, adult_pub_dists,
                            os.path.join(data_dir, "adulthood_pubertal_ids_distribution.csv"))
    export_distribution_csv(adult_iters, adult_adult_dists,
                            os.path.join(data_dir, "adulthood_adult_ids_distribution.csv"))

    all_pub_ids = set()
    for dist_dict in puberty_dists:
        all_pub_ids.update(dist_dict.keys())
    for dist_dict in adult_pub_dists:
        all_pub_ids.update(dist_dict.keys())
    all_pub_ids = sorted(all_pub_ids)

    stack_plot_path = os.path.join(figures_dir, "puberty_phase_pubertal_ids.png")
    plot_stack_counts_with_fixed_ids(
        iterations=puberty_iters,
        dist_series=puberty_dists,
        all_ids=all_pub_ids,
        title="Puberty Phase (Pubertal IDs)",
        output_filename=stack_plot_path,
        dpi=300
    )

    # Calculate average cells of pubertal IDs over time
    avg_pubertal_cells = []
    for dist_dict in puberty_dists:
        if len(dist_dict) == 0:
            avg_pubertal_cells.append(0)
            continue
        total_pubertal = sum(dist_dict.values())
        avg_pubertal_cells.append(total_pubertal / len(dist_dict))



    from matplotlib import cm
    cmap = cm.get_cmap("tab20")
    pubertal_color_map = [cmap(i % 20) for i in range(len(all_pub_ids))]

    stack_plot_path_adult_pub = os.path.join(figures_dir, "adult_phase_pubertal_ids.png")
    plot_stack_counts_with_fixed_ids(
        iterations=adult_iters,
        dist_series=adult_pub_dists,
        all_ids=all_pub_ids,
        title="Adult Phase (Pubertal IDs)",
        output_filename=stack_plot_path_adult_pub,
        color_map=pubertal_color_map,
        dpi=300
    )

    all_adult_ids = set()
    for dist_dict in adult_adult_dists:
        all_adult_ids.update(dist_dict.keys())


    plt.figure(figsize=(8, 5))
    plt.plot(adult_iters, [len(d) for d in adult_adult_dists],
             label="Unique Adult IDs", color="red")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of Unique Adult IDs")
    plt.title("Unique Adult IDs Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "unique_adult_ids_over_time.png"), dpi=300)
    plt.close()

    # Plot unique pubertal IDs in adulthood
    plt.figure(figsize=(8, 5))
    plt.plot(adult_iters, [len(d) for d in adult_pub_dists],
             label="Unique Pubertal IDs", color="blue")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of Unique Pubertal IDs in total duct system")
    plt.title("Unique Pubertal IDs Over Time in total duct")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "unique_pubertal_ids_over_time.png"), dpi=300)
    plt.close()

    # Active TEBs over time
    plt.figure(figsize=(8, 5))
    plt.plot(puberty_iters, num_active_tebs, label="Active TEBs", color="green")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of TEBs")
    plt.title("Active TEBs Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "active_tebs_over_time.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------------------------------
    # 9) Average stem cells per adult ID
    # ----------------------------------------------------------------------------
    avg_stem_cells = []
    for dist_dict in adult_adult_dists:
        if len(dist_dict) == 0:
            avg_stem_cells.append(0)
            continue
        total_stem = sum(dist_dict.values())
        avg_stem_cells.append(total_stem / len(dist_dict))

    plt.figure(figsize=(8, 5))
    plt.plot(adult_iters, avg_stem_cells, label="Average Stem Cells per Adult ID", color="blue")
    plt.xlabel("Time (days)")
    plt.ylabel("Average Stem Cells")
    plt.title("Average Stem Cells per Unique Adult ID Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "avg_stem_cells_per_adult_id.png"), dpi=300)
    plt.close()

    unique_clones_per_duct = progress_data_adult["unique_clones_per_duct"]  # list of dicts
    average_clones_per_duct = []
    for d in unique_clones_per_duct:
        if len(d) == 0:
            average_clones_per_duct.append(0)
        else:
            average_clones_per_duct.append(sum(d.values()) / len(d))


    plt.figure(figsize=(8, 5))
    plt.plot(adult_iters, average_clones_per_duct, label="Avg Unique Clones (Pub IDs) per Duct", color="purple")
    plt.xlabel("Time (days)")
    plt.ylabel("Average Clones per Duct")
    plt.title("Average Unique Pubertal IDs per Duct Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "avg_unique_pubertal_ids_per_duct.png"), dpi=300)
    plt.close()


    # Combine it with the average stem cells per duct and plot

    plt.figure(figsize=(5, 5))
    # we divide puberty_iters by 10 to relate it to adult MaSCs
    plt.plot(puberty_iters, [x/10 for x in avg_pubertal_cells], label="Adult MaSCs per Pubertal clone", color="red")
    plt.plot(adult_iters, avg_stem_cells, label="Adult MaSCs per Adult clone", color="blue")
    plt.yscale("log")
    plt.xlabel("Time (days)")
    plt.ylabel("Average Stem Cells")
    plt.title("Average Cells per unique clone Over Time")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "avg_cells_per_unique_clone_over_time.png"), dpi=300)


    unique_clones_csv = os.path.join(data_dir, "unique_clones_per_duct.csv")
    with open(unique_clones_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "duct_id", "unique_pubertal_ids"])
        for iteration_val, duct_dict in zip(adult_iters, unique_clones_per_duct):
            for duct_id, unique_count in duct_dict.items():
                writer.writerow([iteration_val, duct_id, unique_count])
    total_cells_after_puberty = sum([d for d in puberty_dists[-1].values()])
    total_stem_cells_after_adulthood = sum([d for d in adult_pub_dists[-1].values()])

    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, "w") as readme_file:
        readme_text = f"""# Ductal Tree Simulation Outputs


This folder contains outputs from the ductal tree simulation in two phases:
Puberty and Adulthood.

## Simulation Parameters
- total number of cells after puberty: {total_cells_after_puberty}
- total number of stem cells after adulthood: {total_stem_cells_after_adulthood}
- bifurcation_prob: {bifurcation_prob}
- initial_side_count: {n_clones/2}
- initial_center_count: {n_clones/2}
- initial_termination_prob: {initial_termination_prob}
- adulthood rounds: 33
- defined time of puberty: {puberty_days} days
- defined time of adulthood: {adult_days} days

## Statistics after simulation:
- Average number of cells per duct: {total_cells_after_puberty / G_copy.number_of_edges()}
- Average number of stem cells per duct: {total_stem_cells_after_adulthood / G.number_of_edges()}


## Figures
All figures are saved in the `figures` directory:
1. ductal_tree_after_puberty.png: The ductal tree, color coded by if the duct contains a selected ID. Before adulthood.
2. ductal_tree_after_adulthood.png: The ductal tree, color coded by if the duct contains a selected ID. After adulthood.
3. puberty_phase_pubertal_ids.png: Stacked area plot of Pubertal IDs over time, during pubertal development.
4. adult_phase_pubertal_ids.png: Stacked area plot of Pubertal IDs over time, during adulthood.
5. unique_adult_ids_over_time.png: Number of unique adult IDs over time.
6. unique_pubertal_ids_over_time.png: Number of unique pubertal IDs in the total duct system over time.
7. active_tebs_over_time.png: Number of active TEBs over time during puberty.
8. avg_stem_cells_per_adult_id.png: Average number of stem cells per remaining unique adult ID over time during adulthood.
9. avg_unique_pubertal_ids_per_duct.png: Average number of unique pubertal IDs per duct over time during adulthood.
10. avg_cells_per_unique_clone_over_time.png: Average number of cells per unique clone over time during puberty and adulthood.

## CSV Data
All CSV files are in the `csv_data` directory:
It is a bit clunky to read, if the data from the plots is needed, please ask.

1. puberty_pubertal_ids_distribution.csv
   - Columns: iteration, clone_id, count
   - Distribution of Pubertal IDs during puberty phase

2. adulthood_pubertal_ids_distribution.csv
   - Columns: iteration, clone_id, count
   - Distribution of Pubertal IDs during adulthood

3. adulthood_adult_ids_distribution.csv
   - Columns: iteration, clone_id, count
   - Distribution of Adult IDs during adulthood

4. unique_clones_per_duct.csv
   - Columns: iteration, duct_id, unique_pubertal_ids
   - Number of unique pubertal IDs per duct at each iteration in adulthood

## Notes
- The random seed is set to {seed} for reproducibility.
"""
        readme_file.write(readme_text)

    print(f"Simulation completed. Outputs are in '{output_dir}'.")


if __name__ == "__main__":
    main()
