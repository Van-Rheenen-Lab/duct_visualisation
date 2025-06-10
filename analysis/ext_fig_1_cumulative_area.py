from analysis.utils.cumulative_area import (analyze_area_vs_branch_level_multi_corridor,
                                            plot_area_vs_branch_level_multi_stack, plot_pixel_levels,
                                            mean_label_fraction_by_level, sliding_window_label_fraction)
import json
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage import io
import matplotlib.pyplot as plt
from shapely.validation import make_valid
from analysis.utils.loading_saving import load_duct_systems, create_directed_duct_graph, find_root, select_biggest_duct_system
import networkx as nx
import numpy as np

if __name__ == "__main__":
    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

    green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
    yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
    red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
    threshold_value = 400

    ## BhomPhet
    # json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\890_annotations.json'
    # duct_borders_path =  r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max.lif - TileScan 1 Merged_Processed001_duct.geojson'
    # red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max_forbranchanalysis-0003.tif'
    # yellow_image_path = None
    # green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max_forbranchanalysis-0001.tif'
    # threshold_value = 400

    # # # Phet
    # red_image_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\28052024_2435322_L5_ecad_mAX-0006.tif'
    # duct_borders_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\annotations_exported.geojson'
    # json_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'
    # green_image_path = None
    # yellow_image_path = None
    # threshold_value = 400

    ## BhomPhom
    # json_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2437324_BhomPhom_24W\annotations 7324-1.json"
    # duct_borders_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2437324_BhomPhom_24W\01062024_7324_L5_sma_max.lif - TileScan 1 Merged_Processed001_forbranchanalysis.geojson"
    # red_image_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2437324_BhomPhom_24W\30052024_7324_L5_sma_max_clean_forbranchanalysis-0005.tif"
    # green_image_path = None
    # yellow_image_path = None
    # threshold_value = 400

    def safe_read(path):
        return io.imread(path) if path else None

    red_image = safe_read(red_image_path)
    green_image = safe_read(green_image_path)
    yellow_image = safe_read(yellow_image_path)
    images = [red_image, green_image, yellow_image]
    channel_labels = ["Red", "Green", "Yellow"]

    # Load duct system data
    duct_systems = load_duct_systems(json_path)
    duct_system = select_biggest_duct_system(duct_systems)
    G_dir = create_directed_duct_graph(duct_system)

    root_node = find_root(G_dir)

    # Load and fix duct borders from geojson.
    with open(duct_borders_path, 'r') as f:
        duct_borders = json.load(f)
    valid_geoms = []
    for feat in duct_borders['features']:
        geom = shape(feat['geometry'])
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_valid:
            valid_geoms.append(geom)
    duct_polygon = unary_union(valid_geoms)

    # # Run branch-level corridor analysis without external duct_system.
    # area_by_lvl, positives_by_lvl, pixel_levels = analyze_area_vs_branch_level_multi_corridor(
    #     G_dir=G_dir,
    #     root_node=root_node,
    #     duct_polygon=duct_polygon,
    #     images=images,
    #     threshold=threshold_value,
    #     buffer_dist=10.0
    # )
    #
    # # Plot stacked cumulative area vs. branch level.
    # plot_area_vs_branch_level_multi_stack(
    #     area_by_lvl,
    #     positives_by_lvl,
    #     channel_labels=channel_labels,
    #     use_log=False
    # )
    # plt.savefig("stacked_plot.png", dpi=600)

    # # Show pixel-level branch assignment.
    # max_level = len(area_by_lvl) - 1
    # plot_pixel_levels(pixel_levels, max_level)
    # plt.savefig("pixel_levels.png", dpi=600)
    #
    # node_levels = dict(nx.shortest_path_length(G_dir, source=root_node))
    # edge_levels = [node_levels[child] for _, child in G_dir.edges()]
    #
    # max_edge_level = max(edge_levels, default=0)
    # bins = range(max_edge_level + 2)
    #
    # plt.figure(figsize=(8, 5))
    # plt.hist(edge_levels, bins=bins, align='left', rwidth=0.8)
    # plt.xlabel("Branch level")
    # plt.ylabel("Number of duct segments")
    # plt.title("Distribution of duct segments per branch level")
    # # plt.savefig("branch_histogram.png", dpi=600)
    #
    # mean_frac_by_lvl = mean_label_fraction_by_level(
    #     G_dir=G_dir,
    #     root_node=root_node,
    #     duct_polygon=duct_polygon,
    #     images=images,
    #     buffer_dist=10.0,  # keep identical to earlier corridor width
    #     threshold=threshold_value
    # )
    #
    # levels = np.arange(len(mean_frac_by_lvl))
    # valid = ~np.isnan(mean_frac_by_lvl)  # skip levels with no labelled ducts
    #
    # plt.figure(figsize=(6, 4))
    # plt.plot(levels[valid], mean_frac_by_lvl[valid], marker='o')
    # plt.xlabel("Branch level")
    # plt.ylabel("Average fraction of labelled clone")
    # plt.title("Labelling density across branch levels")
    # plt.tight_layout()
    # # plt.savefig("label_density_vs_level.png", dpi=600)

    # --- sliding‐window label fraction plot ---
    window_size = 5
    fractions_by_channel, _, _ = sliding_window_label_fraction(
        G_dir=G_dir,
        root_node=root_node,
        duct_polygon=duct_polygon,
        images=images,
        threshold=threshold_value,
        buffer_dist=10.0,
        window_size=window_size
    )

    plt.figure(figsize=(6, 4))
    for ch, data in fractions_by_channel.items():
        if not data:
            continue
        # unpack levels and fractions
        levels, fracs = zip(*data)
        plt.plot(levels, fracs, marker='o', label=channel_labels[ch])
    plt.xlabel("Branch level (window end)")
    plt.ylabel(f"Label fraction (window size={window_size})")
    plt.title("Sliding-window Label Fraction by Branch Level")
    plt.legend(loc="best")
    plt.tight_layout()
    # plt.savefig("sliding_window_label_fraction.png", dpi=600)
    # -----------------------------------------


    plt.show()
    print("Analysis complete!")
