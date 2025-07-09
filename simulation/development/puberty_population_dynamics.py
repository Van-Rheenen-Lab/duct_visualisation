import matplotlib.pyplot as plt
import numpy as np
from simulation.puberty_deposit_elimination import simulate_ductal_tree
from simulation.utils.plotting_simulated_ducts import plotting_ducts
import random

def run_sim_and_plot():
    random.seed(41)

    n_clones = 170
    bifurcation_prob = 0.01
    initial_termination_prob = 0.25
    final_termination_prob = 0.55
    max_cells = 3_000_000

    # -- Simulate Puberty --
    G, progress_data = simulate_ductal_tree(
        max_cells=max_cells,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=n_clones / 2,
        initial_center_count=n_clones / 2,
        initial_termination_prob=initial_termination_prob,
        final_termination_prob=final_termination_prob
    )

    # Basic times
    iterations = progress_data["iteration"]
    total_cells = progress_data["total_cells"]  # never decreasing if depositing properly
    clone_counts_series = progress_data["clone_counts_over_time"]  # list of dicts
    num_active_tebs = progress_data["num_active_tebs"]
    avg_dom_fracs = progress_data["avg_dom_fraction"]
    avg_stem_dom_fracs = progress_data["avg_dom_stem_fraction"]

    # 2) Plot total cells over time
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, total_cells, label="Total Cells", color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Total # of Cells")
    plt.title("Total Population Over Time")
    plt.legend()
    plt.tight_layout()


    # 3) Stacked Area Plot of clones over time
    # Identify all unique clones
    all_clone_ids = set()
    for d in clone_counts_series:
        all_clone_ids.update(d.keys())
    all_clone_ids = sorted(all_clone_ids)

    # Build a time series for each clone
    clone_time_series = {clone_id: [] for clone_id in all_clone_ids}
    for d in clone_counts_series:
        for cid in all_clone_ids:
            clone_time_series[cid].append(d.get(cid, 0))

    # Convert to array
    data_matrix = []
    for i in range(len(iterations)):
        row = [clone_time_series[cid][i] for cid in all_clone_ids]
        data_matrix.append(row)
    data_matrix = np.array(data_matrix).T  # shape (#clones, #iterations)

    plt.figure(figsize=(10,6))
    plt.stackplot(iterations, data_matrix, labels=[f"Clone {cid}" for cid in all_clone_ids])
    plt.xlabel("Iteration")
    plt.ylabel("Cell Count")
    plt.title("Stacked Area: Clones Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()



    # 4) Plot the number of active TEBs over time
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, num_active_tebs, label="Active TEBs", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Number of TEBs")
    plt.title("Active TEBs Over Time")
    plt.legend()
    plt.tight_layout()


    # 5) Plot the average dominant clone fraction among TEBs
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, avg_dom_fracs, label="Average Dom Fraction", color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Fraction")
    plt.title("Average Dominant Clone Fraction (stem cells + deposited cells) Over All Active TEBs")
    plt.legend()
    plt.tight_layout()


    # 6) (Optional) Plot each TEB's dominant fraction individually
    teb_history = progress_data["teb_history"]
    plt.figure(figsize=(10, 6))
    for node_id, data_dict in teb_history.items():
        iters = data_dict["iteration"]
        dom_fracs = data_dict["dominant_clone_fraction"]
        plt.plot(iters, dom_fracs, label=f"TEB {node_id}")
    plt.xlabel("Iteration")
    plt.ylabel("Dominant Clone Fraction")
    plt.title("Dominant Clone Fraction (Stem + Deposited Cells) Per TEB Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()


    teb_history = progress_data["teb_history"]
    plt.figure(figsize=(10, 6))
    for node_id, data_dict in teb_history.items():
        iters = data_dict["iteration"]
        dom_fracs = data_dict["dominant_stem_cell_fraction"]
        plt.plot(iters, dom_fracs, label=f"TEB {node_id}")
    plt.xlabel("Iteration")
    plt.ylabel("Dominant Clone Fraction")
    plt.title("Dominant Clone Fraction (Stem) Per TEB Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 5) Plot the average dominant clone fraction among TEBs
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, avg_stem_dom_fracs, label="Average Dom Fraction stem cells", color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Fraction")
    plt.title("Average Dominant Clone Fraction (stem cells) Over All Active TEBs")
    plt.legend()
    plt.tight_layout()

    plotting_ducts(G)
    # turn of legend
    plt.legend().set_visible(False)

    # 7) Plot the pre-computed unique-stem-cell count
    unique_stem_counts = progress_data["unique_stem_counts"]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations,
             unique_stem_counts,
             label="Unique stem cells in active TEBs",
             color="purple")
    plt.xlabel("Iteration")
    plt.ylabel("# Unique stem cells")
    plt.title("Stem-cell diversity across active TEBs")
    plt.legend()
    plt.tight_layout()

    peak_iter_idx = int(np.argmax(num_active_tebs))  # index, not the label
    peak_iter = iterations[peak_iter_idx]  # iteration number
    peak_avg_dom = avg_stem_dom_fracs[peak_iter_idx]

    print(
        f"Iteration with the most active TEBs: {peak_iter} "
        f"(#TEBs = {num_active_tebs[peak_iter_idx]})\n"
        f"  ↳ Average dominant-clone fraction (stem cells) = {peak_avg_dom:.3f}"
    )

    plt.show()
    pass
if __name__ == "__main__":
    run_sim_and_plot()
