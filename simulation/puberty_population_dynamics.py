# analysis_script.py
import matplotlib.pyplot as plt
import numpy as np
from puberty import simulate_ductal_tree
from analysis.utils.plotting_trees import plot_hierarchical_graph, create_annotation_color_map
import random

def run_sim_and_plot():
    random.seed(42)

    # 1) Run simulation
    G, progress_data = simulate_ductal_tree(
        max_cells=9_000_000,
        bifurcation_prob=0.01,
        initial_side_count=50,
        initial_center_count=50,
        initial_termination_prob=0.3
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
    plt.show()
    pass

    # for
if __name__ == "__main__":
    run_sim_and_plot()
