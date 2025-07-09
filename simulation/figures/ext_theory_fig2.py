import os
import json
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geom
import shapely.ops as ops
from shapely.validation import make_valid
from rasterio.features import rasterize
from skimage import io

from analysis.utils.loading_saving import (
    load_duct_systems,
    create_directed_duct_graph,
    find_root,
    select_biggest_duct_system,
)
from analysis.utils.cumulative_area import compute_branch_levels
from analysis.utils.plotting_striped_trees import plot_hierarchical_graph_subsegments
from analysis.utils.plotting_striped_trees import plot_hierarchical_graph_subsegments as plot_real
from simulation.utils.plotting_simulated_ducts_striped import (
    plot_hierarchical_graph_subsegments_simulated as plot_sim,
)
from simulation.puberty_deposit_elimination import simulate_ductal_tree
import random


# helpers -----------------------------------------------------------------
def branch_level_counts(g, root):
    levels = compute_branch_levels(g, root)
    return np.bincount(list(levels.values()))


def build_duct_mask(borders_path, base_shape):
    with open(borders_path, "r") as f:
        borders = json.load(f)
    valid = []
    for feat in borders["features"]:
        geom_obj = geom.shape(feat["geometry"])
        if not geom_obj.is_valid:
            geom_obj = make_valid(geom_obj)
        if geom_obj.is_valid:
            valid.append(geom_obj)
    polygon = ops.unary_union(valid)
    shapes = [(polygon, 1)]
    return rasterize(shapes, out_shape=base_shape, fill=0, dtype=np.uint8)


def save_striped_tree(
    g,
    root,
    filename,
    simulated=False,
    duct_mask=None,
    red_image=None,
    green_image=None,
    yellow_image=None,
    threshold=300,
):
    plot_fn = plot_sim if simulated else plot_real
    kw = dict(
        root_node=root,
        use_hierarchy_pos=True,
        orthogonal_edges=True,
        vert_gap=3.5,
        draw_nodes=False,
        linewidth=0.9 if simulated else 0.6,
    )
    if not simulated:
        kw.update(
            duct_mask=duct_mask,
            red_image=red_image,
            green_image=green_image,
            yellow_image=yellow_image,
            threshold=threshold,
        )
    fig, _ = plot_fn(g, **kw)
    fig.set_size_inches(35, 12)
    fig.savefig(filename, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_branch_hist(counts, title, color, filename):
    x = np.arange(counts.size)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, counts, color = color)
    ax.set_xlabel("branch level")
    ax.set_ylabel("number of ducts")
    # ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, format="svg", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    random.seed(41)

    # plotting style
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "Arial",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    output_folder = "output_svgs"
    os.makedirs(output_folder, exist_ok=True)

    # real datasets --------------------------------------------------------
    datasets = [
        dict(
            name="2473536_Cft_24W",
            json_path=r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json",
            borders_path=r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson",
            green_image=r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif",
            yellow_image=r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif",
            red_image=r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif",
            threshold=300,
        ),
        dict(
            name="2435322_Phet_24W",
            json_path=r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2435322_Phet_24W\clean_and_expanded.json",
            borders_path=r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2435322_Phet_24W\28052024_2435322_L5_ecad_mAX.lif - TileScan 2 Merged_Processed001_outlines.geojson",
            green_image=None,
            yellow_image=None,
            red_image=r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2435322_Phet_24W\28052024_2435322_L5_ecad_mAX_hyperstackforbranchanalysis-0003.tif",
            threshold=300,
        ),
    ]

    for d in datasets:
        ducts = load_duct_systems(d["json_path"])
        g_real = create_directed_duct_graph(select_biggest_duct_system(ducts))
        root = find_root(g_real)

        red_im = io.imread(d["red_image"]) if d["red_image"] else None
        base_shape = red_im.shape if red_im is not None else (1024, 1024)
        duct_mask = build_duct_mask(d["borders_path"], base_shape)

        save_striped_tree(
            g_real,
            root,
            os.path.join(output_folder, f"{d['name']}_tree.svg"),
            simulated=False,
            duct_mask=duct_mask,
            red_image=red_im,
            green_image=io.imread(d["green_image"]) if d["green_image"] else None,
            yellow_image=io.imread(d["yellow_image"]) if d["yellow_image"] else None,
            threshold=d["threshold"],
        )
        save_branch_hist(
            branch_level_counts(g_real, root),
            f"{d['name']} branch-level distribution",
            "blue",
            os.path.join(output_folder, f"{d['name']}_hist.svg"),

        )

    # simulated puberty tree ----------------------------------------------
    sim_kwargs = dict(
        max_cells=3_000_000,
        bifurcation_prob=0.01,
        initial_side_count=85,
        initial_center_count=85,
        initial_termination_prob=0.25,
        final_termination_prob=0.55,
    )

    g_sim, _ = simulate_ductal_tree(**sim_kwargs)
    root_sim = next(iter(g_sim.nodes))

    save_striped_tree(
        g_sim,
        root_sim,
        os.path.join(output_folder, "simulation_tree.svg"),
        simulated=True,
    )
    save_branch_hist(
        branch_level_counts(g_sim, root_sim),
        "simulated branch-level distribution",
        "red",
        os.path.join(output_folder, "simulation_hist.svg"),
    )
