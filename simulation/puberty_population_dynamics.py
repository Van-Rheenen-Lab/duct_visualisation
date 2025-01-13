# analysis_script.py
import matplotlib.pyplot as plt
import numpy as np
from puberty import simulate_ductal_tree
from analysis.utils.plotting_trees import plot_hierarchical_graph, create_annotation_color_map
import random

def plotting_ducts(G):

    # 2) Build the system_data structure
    system_data = {"segments": {}}

    # Loop over edges, read out duct_clones to decide how to annotate
    for (u, v) in G.edges():
        segment_name = f"duct_{u}_to_{v}"
        duct_clones = G[u][v].get("duct_clones", [])

        # Optionally set an annotation if certain clones appear
        ann = None
        if 42 in duct_clones:
            ann = "42"
        elif 84 in duct_clones:
            ann = "90"
        elif 12 in duct_clones:
            ann = "12"
        elif 24 in duct_clones:
            ann = "24"

        if ann:
            system_data["segments"][segment_name] = {
                "properties": {"Annotation": ann}
            }
        else:
            system_data["segments"][segment_name] = {
                "properties": {}
            }

        # Store segment name for reference
        G[u][v]["segment_name"] = segment_name

    # Create a color map based on any annotations
    color_map = create_annotation_color_map(system_data, colormap_name='tab10')

    # 3) Plot the tree with your hierarchical graph function
    fig, ax = plot_hierarchical_graph(
        G,
        system_data=system_data,
        annotation_to_color=color_map,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=2,
        linewidth=1.5
    )

    # 4) Calculate the average number of cells per segment
    total_cells = 0
    edge_count = 0

    for (u, v) in G.edges():
        edge_clones = G[u][v].get("duct_clones", [])
        total_cells += len(edge_clones)
        edge_count += 1

    if edge_count > 0:
        avg_cells = total_cells / edge_count
    else:
        avg_cells = 0

    print(f"Average number of cells per segment: {avg_cells}")


def run_sim_and_plot():
    random.seed(42)

    # 1) Run simulation
    G, progress_data = simulate_ductal_tree(
        max_cells=12_000_00,
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
