from simulation.puberty import simulate_ductal_tree_on_existing_graph
from analysis.utils.loading_saving import load_duct_systems, create_directed_duct_graph
from analysis.utils.fixing_annotations import simplify_graph
import matplotlib.pyplot as plt
import numpy as np
from simulation.utils.plotting_simulated_ducts import plotting_ducts
import random

""" 
The following script was made 
"""

if __name__ == "__main__":
    random.seed(42)

    # json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    # json_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'
    json_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_carmine\06012025_carminegood images\export files\2516525-slide1_9weeks_branch.json"
    # json_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'
    graphs = load_duct_systems(json_path)

    big_graph_len = 0
    for i, g in enumerate(graphs):
        if len(graphs[g]["branch_points"]) > big_graph_len:
            big_graph_len = len(graphs[g]["branch_points"])
            index = i


    G = create_directed_duct_graph(graphs[index])
    print(f"Using index {index} as the graph")

    first_bp = list(G.nodes)[0]
    while len(list(G.predecessors(first_bp))) == 1:
        first_bp = list(G.predecessors(first_bp))[0]
    root_node = first_bp
    print(f"Using branch point {root_node} as BFS root")

    G = create_directed_duct_graph(graphs[index])

    G = simplify_graph(G)
    G, progress_data = simulate_ductal_tree_on_existing_graph(
        existing_graph=G,
        root_node=root_node,
        bifurcation_prob=0.01,
        replacement_prob=1,
        initial_side_count=85,
        initial_center_count=85,
    )

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

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, num_active_tebs, label="Active TEBs", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Number of TEBs")
    plt.title("Active TEBs Over Time")
    plt.legend()
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

    plotting_ducts(G, root_node=root_node)

    # save picture at 600 dpi with the name 2516525-slide1_9weeks_branch
    plt.savefig('2516525-slide1_9weeks_branch.png', dpi=600)

    import networkx as nx

    node_levels = dict(nx.shortest_path_length(G, source=root_node))

    clones_per_level = {}
    for parent, child, data in G.edges(data=True):
        level = node_levels.get(child)
        clone_ids = data.get("duct_clones", [])
        unique_count = len(set(clone_ids))
        clones_per_level.setdefault(level, []).append(unique_count)

    levels = sorted(clones_per_level)
    avg_unique_stems = [np.mean(clones_per_level[L]) for L in levels]

    # Delete the first 10 levels
    levels = levels[10:]
    avg_unique_stems = avg_unique_stems[10:]

    plt.figure(figsize=(6, 4))
    plt.plot(levels, avg_unique_stems, marker="o")
    plt.xlabel("Branch level")
    plt.ylabel("Average # progenitor MaSCs per duct segment")
    plt.title("Clonal diversity across branch levels")
    plt.tight_layout()

    plt.show()