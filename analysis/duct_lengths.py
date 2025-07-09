import numpy as np
import matplotlib.pyplot as plt

def _segment_length_2d(seg_data, bp_lookup):
    """Return 2-D length of one segment, including internal_points."""
    # build ordered coordinate lists
    xs = [bp_lookup[seg_data["start_bp"]][0]]
    ys = [bp_lookup[seg_data["start_bp"]][1]]

    for p in seg_data.get("internal_points", []):
        xs.append(p["x"])
        ys.append(p["y"])

    xs.append(bp_lookup[seg_data["end_bp"]][0])
    ys.append(bp_lookup[seg_data["end_bp"]][1])

    dx = np.diff(xs)
    dy = np.diff(ys)
    return np.hypot(dx, dy).sum()


def segment_lengths_2d(duct_system):
    """
    Return a NumPy array of 2-D segment lengths for one duct system.
    """
    bp_lookup = {name: (d["x"], d["y"])            # fast coordinate lookup
                 for name, d in duct_system["branch_points"].items()}

    lengths = [
        _segment_length_2d(seg, bp_lookup)
        for seg in duct_system["segments"].values()
    ]
    return np.asarray(lengths)

def cells_per_duct(G, *, include_empty=False):
    """Return an array with #cells for every edge of the graph G."""
    counts = [
        len(data.get("duct_clones", []))
        for _, _, data in G.edges(data=True)
        if include_empty or data.get("duct_clones")
    ]
    return np.asarray(counts, dtype=int)

def plot_distribution(vals, *, bins="auto", title="Cells per duct"):
    fig, ax = plt.subplots()
    ax.hist(vals, bins=bins, color="red")
    ax.set_xlabel("Cells")
    ax.set_ylabel("Number of ducts")
    # ax.set_title(title)
    return fig



if __name__ == "__main__":
    from analysis.utils.loading_saving import load_duct_systems, select_biggest_duct_system, create_directed_duct_graph
    from simulation.puberty_deposit_elimination import simulate_ductal_tree

    plt.rcParams.update({'font.size': 14})  # set plotting font size

    json_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json"

    duct_systems = load_duct_systems(json_path)
    duct_system = select_biggest_duct_system(duct_systems)
    G = create_directed_duct_graph(duct_system)

    lengths_pix = segment_lengths_2d(duct_system)
    pixel_to_micron = 1.2133
    lengths = lengths_pix * pixel_to_micron  # convert to microns

    fig, ax = plt.subplots()
    ax.hist(lengths, bins="auto", color="blue")
    ax.set_xlabel("Segment length (µm)")    # change unit label if necessary
    ax.set_ylabel("Number of ducts")
    ax.set_yscale("log")
    ax.set_xlim(left=0, right=1200)
    # ax.set_title("Distribution of duct-segment lengths")
    fig.tight_layout()
    # save as SVG
    fig.savefig("duct_segment_lengths.svg", format="svg", bbox_inches="tight")

    import random

    random.seed(41)

    n_clones = 170
    bifurcation_prob = 0.01
    initial_termination_prob = 0.25
    final_termination_prob = 0.55
    max_cells = 3_000_000

    # -- Simulate Puberty --
    G_sim, progress_data = simulate_ductal_tree(
        max_cells=max_cells,
        bifurcation_prob=bifurcation_prob,
        initial_side_count=n_clones / 2,
        initial_center_count=n_clones / 2,
        initial_termination_prob=initial_termination_prob,
        final_termination_prob=final_termination_prob
    )

    cells_per_duct_sim = cells_per_duct(G_sim, include_empty=True)
    fig2 = plot_distribution(cells_per_duct_sim, bins=100, title="Cells per duct (simulated)")
    ax2 = fig2.gca()
    ax2.set_yscale("log")
    ax2.set_xlim(left=0, right=8000)

    fig2.tight_layout()
    fig2.savefig("cells_per_duct_simulated.svg", format="svg", bbox_inches="tight")

    # Number of 0s in cells_per_duct_sim

    plt.show()

    print(f"{len(lengths)} segments analysed")
    print(f"mean = {lengths.mean():.2f} µm,  median = {np.median(lengths):.2f} µm")
    print(f"simulated ducts: {len(cells_per_duct_sim)}")
    print(f"simulated mean = {cells_per_duct_sim.mean():.2f} cells,  median = {np.median(cells_per_duct_sim):.2f} cells")

