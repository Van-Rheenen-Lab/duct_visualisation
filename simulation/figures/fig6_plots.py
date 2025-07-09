import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation.puberty_deposit_elimination import simulate_ductal_tree
from simulation.adulthood import simulate_adulthood

def main():
    # Set up the plotting style
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'Arial',
        'figure.figsize': (8, 6)
    })

    # Set parameters and directories
    seed = 41
    random.seed(seed)
    n_clones = 170
    output_dir = "simulation_outputs"
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    bifurcation_prob = 0.01
    initial_termination_prob = 0.25

    # --- Run Puberty Simulation ---
    G, progress_data_puberty = simulate_ductal_tree(
        max_cells=3_000_000,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=n_clones / 2,
        initial_center_count=n_clones / 2,
        initial_termination_prob=initial_termination_prob,
        final_termination_prob=0.55
    )

    # Scale puberty iterations to a 4-week period (4*7 days = 28 days)
    puberty_iters = progress_data_puberty["iteration"]
    puberty_days = 4 * 7  # 28 days
    puberty_iters = [(i / max(puberty_iters)) * puberty_days for i in puberty_iters]
    puberty_dists = progress_data_puberty["clone_counts_over_time"]

    # Compute average pubertal cell counts per iteration from puberty simulation
    avg_pubertal_cells = []
    for dist in puberty_dists:
        if len(dist) == 0:
            avg_pubertal_cells.append(0)
        else:
            avg_pubertal_cells.append(sum(dist.values()) / len(dist))

    # --- Run Adulthood Simulation ---
    G, progress_data_adult = simulate_adulthood(G, rounds=33+17)
    adult_cycles = progress_data_adult["iteration"]
    adult_days = 33 * 4.5  # each round is 4.5 days
    adult_iters = [i * 4.5 for i in adult_cycles]

    adult_pub_dists = progress_data_adult["pubertal_id_counts"]
    adult_adult_dists = progress_data_adult["adult_id_counts"]

    # Compute average adult stem cell counts per iteration from adult IDs (used for the "Adult clones" plot)
    avg_stem_cells = []
    for dist in adult_adult_dists:
        if len(dist) == 0:
            avg_stem_cells.append(0)
        else:
            avg_stem_cells.append(sum(dist.values()) / len(dist))

    # --- Define Consistent Colors ---
    adult_color = 'blue'
    pubertal_color = 'red'

    # --- Update Pubertal Clone Curve in Adulthood Phase ---
    # 1. Use the final pubertal clone value from the puberty simulation as the baseline
    pubertal_base = avg_pubertal_cells[-1] / 10

    # 2. Calculate total adult MaSCs at each adult timepoint (from the adult simulation)
    adult_total_mascs = []
    for dist in adult_adult_dists:
        adult_total_mascs.append(sum(dist.values()) if len(dist) > 0 else 0)

    # 3. Determine the number of surviving unique pubertal clones at each adult timepoint
    unique_pubertal_counts = [len(d) for d in adult_pub_dists]

    # 4. For each timepoint, compute the ratio: (total adult mascs) / (number of surviving pubertal clones)
    ratio_values = []
    for total, count in zip(adult_total_mascs, unique_pubertal_counts):
        if count == 0:
            ratio_values.append(0)
        else:
            ratio_values.append(total / count)

    # 6. Shift the adult timepoints by the maximum puberty time for a continuous time axis
    max_puberty_time = max(puberty_iters)
    adult_iters_offset = [t + max_puberty_time for t in adult_iters]
    # truncate at the end of the adult simulation
    adult_iters_offset = [t for t in adult_iters_offset if t <= max(adult_iters)]
    # also make ratio values shorter
    ratio_values = ratio_values[:len(adult_iters_offset)]

    # as the first value, add the last point of puberty_iters so the line is continuous
    ratio_values = [pubertal_base] + ratio_values
    adult_iters_offset = [adult_iters_offset[0]] + adult_iters_offset

    # --- Plot: Simulated MaSCs per Surviving Clone ---
    plt.figure(figsize=(5, 5))
    # Plot the puberty phase (converting pubertal cell count to adult MaSCs using division by 10)
    plt.plot(puberty_iters, [x / 10 for x in avg_pubertal_cells], color=pubertal_color)
    # Plot the extended pubertal clone values during the adulthood phase
    plt.plot(adult_iters_offset, ratio_values,
             label="Pubertal", color=pubertal_color)
    # Plot the adult clones (using the same offset)
    plt.plot(adult_iters, avg_stem_cells,
             label="Adult", color=adult_color)

    def imaging_point(ax, x, y):
        """
        Put a big 'X' at (x, y) and attach a short text label.
        """
        ax.scatter(x, y, marker='X', s=80, color='black', zorder=10)

    pub_13 = np.interp(13, puberty_iters, [v / 10 for v in avg_pubertal_cells])
    pub_167 = np.interp(167, adult_iters_offset, ratio_values)

    adult_63 = np.interp(63, adult_iters, avg_stem_cells)
    adult_224 = np.interp(224, adult_iters, avg_stem_cells)

    ax = plt.gca()  # current axes
    imaging_point(ax, 13, pub_13)
    imaging_point(ax, 167, pub_167)

    imaging_point(ax, 63, adult_63)
    imaging_point(ax, 224, adult_224)

    plt.yscale("log")
    plt.ylim(bottom=1)  # Ensure y-axis starts at 1
    plt.xlabel("Time (days)")
    plt.ylabel("Adult MaSCs per initial clone")
    plt.title("Simulated MaSCs per Surviving Clone")
    plt.ylim(bottom=0.9, top=3500)
    # # add "imaging point" for the black x in the legend. we've already made them
    plt.scatter([], [], marker='X', s=80, color='black', label='Imaging')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, "increased_adult_mascs_over_time.svg"))
    plt.close()

    # --- Plot: Simulated Clone Survival in Adulthood ---
    unique_pub_ids = [len(d) for d in adult_pub_dists]
    unique_adult_ids = [len(d) for d in adult_adult_dists]
    # Normalize by the value at adult timepoint 0 (avoid division by zero)
    print(unique_pub_ids[0])
    norm_pub = [val / unique_pub_ids[0] if unique_pub_ids[0] else 0 for val in unique_pub_ids]
    norm_adult = [val / unique_adult_ids[0] if unique_adult_ids[0] else 0 for val in unique_adult_ids]

    # only until day 161
    adult_cycles = [t for t in adult_cycles if t <= 33]
    norm_pub = norm_pub[:len(adult_cycles)]
    norm_adult = norm_adult[:len(adult_cycles)]


    plt.figure(figsize=(5, 5))
    plt.plot(adult_cycles, norm_pub, label="Pubertal", color=pubertal_color)
    plt.plot(adult_cycles, norm_adult, label="Adult", color=adult_color)
    plt.xlabel("Estrous cycles")
    plt.ylim(0, 1.2)  # Set survival y-axis from 0 to 1
    plt.ylabel("Surviving clones")
    plt.title("Simulated Initial Clone Survival")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "normalized_ids_over_time.svg"))
    plt.close()

    # Process Experimental Survival Data (from Scheele et al.)
    orig_values = [
        [0.236111111, 0.62037037, 0.416666667, 0.196759259],
        [0.090277778, 0.143518519, 0.085648148, 0.05787037]
    ]
    orig_days = np.array([64, 225])  # (omitting 120)
    orig_means = np.array([np.mean(v) for v in orig_values])
    orig_sems = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) for v in orig_values])
    # Normalize by the first timepoint and shift time to start at 0
    norm_orig_means = orig_means / orig_means[0]
    norm_orig_sems = orig_sems / orig_means[0]
    norm_orig_days = orig_days - orig_days[0]

    new_values = [
        [0, 0, 2, 4, 0],
        [1, 2, 1, 1, 1, 2, 1]
    ]
    new_week_days = np.array([2, 24]) * 7  # Convert weeks to days: [14, 168] (omitting 4)
    new_means = np.array([np.mean(v) for v in new_values])
    new_sems = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) for v in new_values])
    norm_new_means = new_means / new_means[0]
    norm_new_sems = new_sems / new_means[0]
    norm_new_days = new_week_days - new_week_days[0]

    # --- Plot: Experimental Survival Rates (Adult vs Pubertal) ---
    plt.figure(figsize=(7, 5))
    plt.errorbar(norm_new_days, norm_new_means, yerr=norm_new_sems,
                 fmt='s', capsize=5, label='Pubertal',
                 color=pubertal_color)
    plt.errorbar(norm_orig_days, norm_orig_means, yerr=norm_orig_sems,
                 fmt='o', capsize=5, label="Adult (Scheele et al, 2024)",
                 color=adult_color)
    plt.xlabel("Time (days)")
    plt.ylabel("Clone Survival (Normalized)")
    plt.title("Adult vs Pubertal Luminal $Brca^{-/-}P53^{-/-}$ Clone Survival")
    plt.ylim(0, 1.7)  # Set same y-axis range as simulated survival
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "clone_survival_errorbars.png"), dpi=300)
    plt.close()

    # --- Bar Plot of Final Experimental Survival Values ---
    plt.figure(figsize=(4, 4))
    final_values = [norm_orig_means[-1], norm_new_means[-1]]
    final_errors = [norm_orig_sems[-1], norm_new_sems[-1]]
    x_pos = np.arange(len(final_values))
    plt.bar(x_pos, final_values, yerr=final_errors, align='center', alpha=0.7,
            capsize=10, color=[adult_color, pubertal_color])
    plt.xticks(x_pos, ['Adult', 'Pubertal'])
    plt.ylabel('Clone Survival (Normalized)')
    plt.title('Clone Survival after 22/23 weeks')
    plt.ylim(0, 1.7)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "final_timepoint_barplot.png"), dpi=300)
    plt.close()

    # --- Experimental Growth Rate Plot ---
    # Load & aggregate experimental puberty length data (lengths measured per clone)
    exp_puberty_folder = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\FIGURE DATA\lengths CNV files puberty"
    exp_all_sums = []
    for filename in os.listdir(exp_puberty_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(exp_puberty_folder, filename)
            df = pd.read_csv(csv_path, delimiter=";")
            df.rename(columns={
                "Length µm": "Length_um",
                "Length Âµm": "Length_um"
            }, inplace=True)
            group_sums = df.groupby(["Class", "Image"])["Length_um"].sum().reset_index()
            exp_all_sums.append(group_sums)
    exp_puberty_data = pd.concat(exp_all_sums, ignore_index=True)
    exp_puberty_group = exp_puberty_data.groupby(["Image", "Class"])["Length_um"].sum()
    exp_puberty_mean = exp_puberty_group.mean()
    print(exp_puberty_group)
    exp_puberty_sem = np.std(exp_puberty_group.values, ddof=1) / np.sqrt(len(exp_puberty_group))

    # Load experimental adult length data
    # Adult 255 file (semicolon separated) – use dropna() to remove missing values
    adult_255_file = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\FIGURE DATA\lengths adult\adult_pooled_255.csv"
    adult_255_df = pd.read_csv(adult_255_file, header=None, delimiter=";")
    adult_255_values = adult_255_df.dropna().values.flatten()
    exp_adult_255_mean = adult_255_values.mean()
    exp_adult_255_sem = np.std(adult_255_values, ddof=1) / np.sqrt(len(adult_255_values))

    # Adult 64 file (comma separated)
    adult_64_file = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\FIGURE DATA\lengths adult\measurements length clones adult b1p 64.csv"
    adult_64_df = pd.read_csv(adult_64_file, delimiter=',')
    # Use the last column (adjust if needed)
    adult_64_values = adult_64_df[adult_64_df.columns[-1]].values.flatten()
    exp_adult_64_mean = adult_64_values.mean()
    exp_adult_64_sem = np.std(adult_64_values, ddof=1) / np.sqrt(len(adult_64_values))

    # Normalize all experimental growth data to Adult 64 (set Adult 64 = 1)
    norm_exp_adult_64_mean = exp_adult_64_mean / exp_adult_64_mean  # = 1
    norm_exp_adult_255_mean = exp_adult_255_mean / exp_adult_64_mean
    norm_exp_puberty_mean  = exp_puberty_mean  / exp_adult_64_mean

    norm_exp_adult_64_sem = exp_adult_64_sem / exp_adult_64_mean
    norm_exp_adult_255_sem = exp_adult_255_sem / exp_adult_64_mean
    norm_exp_puberty_sem  = exp_puberty_sem / exp_adult_64_mean

    # Define x positions for experimental growth data:
    x_exp_adult64  = 64
    x_exp_puberty  = 255
    x_exp_adult255 = 255

    plt.figure(figsize=(8, 6))
    # Plot Adult 64 point without a label
    plt.errorbar(x_exp_adult64, norm_exp_adult_64_mean, yerr=norm_exp_adult_64_sem,
                 fmt='o', color='blue', label='_nolegend_', capsize=5, markersize=8)
    # Plot Puberty with its label
    plt.errorbar(x_exp_puberty, norm_exp_puberty_mean, yerr=norm_exp_puberty_sem,
                 fmt='s', color='red', label='Puberty', capsize=5, markersize=8)
    # Plot Adult 255 with the label "Adult" (only one adult label)
    plt.errorbar(x_exp_adult255, norm_exp_adult_255_mean, yerr=norm_exp_adult_255_sem,
                 fmt='o', color='blue', label='Adult', capsize=5, markersize=8)
    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Average Length")
    plt.title("Clone lengths")
    plt.yscale("log")
    # Compute y-axis limits so that all error bars are in frame
    ymins = [norm_exp_adult_64_mean - norm_exp_adult_64_sem,
             norm_exp_puberty_mean - norm_exp_puberty_sem,
             norm_exp_adult_255_mean - norm_exp_adult_255_sem]
    ymaxs = [norm_exp_adult_64_mean + norm_exp_adult_64_sem,
             norm_exp_puberty_mean + norm_exp_puberty_sem,
             norm_exp_adult_255_mean + norm_exp_adult_255_sem]
    ymin_val = min(ymins) * 0.9
    ymax_val = max(ymaxs) * 1.1
    plt.ylim(bottom=0.9, top=2500)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "experimental_growth_rate.png"), dpi=300)
    plt.close()

    # also bar a bar plot for the clone lengths final points
    plt.figure(figsize=(4, 4))
    final_values = [norm_exp_adult_255_mean, norm_exp_puberty_mean]
    final_errors = [norm_exp_adult_255_sem, norm_exp_puberty_sem]
    x_pos = np.arange(len(final_values))
    plt.bar(x_pos, final_values, yerr=final_errors, align='center', alpha=0.7,
            capsize=10, color=[adult_color, pubertal_color])
    plt.xticks(x_pos, ['Adult', 'Pubertal'])
    plt.yscale("log")
    plt.ylabel('Average Length (Normalized)')
    plt.title('Clone Lengths after 22/23 weeks')
    plt.ylim(bottom=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "final_timepoint_barplot_lengths.png"), dpi=300)

if __name__ == "__main__":
    main()
