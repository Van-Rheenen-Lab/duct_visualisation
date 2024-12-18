import json
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import LineString, shape
from rasterio.features import rasterize
from skimage import io
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

from utils.fixing_annotations import simplify_duct_system
from utils.plotting_trees import plot_hierarchical_graph
from utils.loading_saving import load_duct_systems, clean_duct_data, create_duct_graph


def compute_segment_percentages(image_path, duct_system, borders_path, threshold):
    if image_path is None:
        return {}, 0, 0

    image = io.imread(image_path)
    with open(borders_path, 'r') as f:
        duct_borders = json.load(f)

    shapes = [(shape(feat['geometry']), 1) for feat in duct_borders['features']]
    mask = rasterize(shapes, out_shape=image.shape, fill=0, dtype=np.uint8, all_touched=False)

    masked_image = image.copy()
    masked_image[mask == 0] = 0
    binary_image = masked_image > threshold

    branch_points = duct_system['branch_points']
    segments = duct_system['segments']

    segment_shapes = []
    segment_id_mapping = {}
    for idx, (seg_name, seg_data) in enumerate(segments.items(), start=1):
        segment_id_mapping[idx] = seg_name
        start_bp = branch_points[seg_data['start_bp']]
        end_bp = branch_points[seg_data['end_bp']]
        pts = [(start_bp['x'], start_bp['y'])]
        pts.extend([(p['x'], p['y']) for p in seg_data.get('internal_points', [])])
        pts.append((end_bp['x'], end_bp['y']))
        line = LineString(pts).buffer(5)
        segment_shapes.append((line, idx))

    segment_mask = rasterize(segment_shapes, out_shape=image.shape, fill=0, dtype=np.uint16, all_touched=True)
    segment_mask[mask == 0] = 0

    valid_indices = segment_mask.flatten() > 0
    flat_segment_mask = segment_mask.flatten()[valid_indices]
    flat_binary = binary_image.flatten()[valid_indices]

    total = np.bincount(flat_segment_mask)
    positive = np.bincount(flat_segment_mask, weights=flat_binary.astype(np.uint8))

    seg_percentages = {}
    max_id = max(segment_id_mapping.keys()) if segment_id_mapping else 0
    for seg_id in range(1, max_id + 1):
        t = total[seg_id] if seg_id < len(total) else 0
        p = positive[seg_id] if seg_id < len(positive) else 0
        seg_percentages[segment_id_mapping[seg_id]] = (p / t * 100) if t > 0 else 0

    # Compute overall percentage
    overall_total = np.sum(total[1:])  # skip index 0 (no segment)
    overall_positive = np.sum(positive[1:])
    overall_percentage = (overall_positive / overall_total * 100) if overall_total > 0 else 0

    return seg_percentages, overall_total, overall_positive


def create_segment_color_map(duct_system, segment_percentages_list):
    num_channels = len(segment_percentages_list)
    colors_list = ['red', 'green', 'blue']
    max_percentages = [max(d.values()) if d else 1 for d in segment_percentages_list]

    cmaps, norms = [], []
    for i in range(num_channels):
        cmap = LinearSegmentedColormap.from_list(f"my_{colors_list[i]}", [(0, 'black'), (1, colors_list[i])])
        cmaps.append(cmap)
        norms.append(colors.Normalize(vmin=0, vmax=max_percentages[i]))

    segments = duct_system['segments']
    segment_color_map = {}
    for seg_name in segments:
        vals = [d.get(seg_name, 0) for d in segment_percentages_list]
        rgb = [int((v / 100) * 255) for v in vals]
        # scale rgb non-linearly
        rgb = [int(255 * (v / 255) ** 0.3) for v in rgb]
        # normalise


        while len(rgb) < 3:
            rgb.append(0)
        segment_color_map[seg_name] = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])

    return segment_color_map, cmaps, norms


def plot_with_channels(duct_system, G, segment_color_map, cmaps, norms, red_used, green_used, blue_used):
    fig, ax = plot_hierarchical_graph(
        G, system_data=duct_system,
        root_node=list(G.nodes)[0],
        use_hierarchy_pos=True,
        vert_gap=8,
        linewidth= 1.1,
        orthogonal_edges=True,
        annotation_to_color={},
        segment_color_map=segment_color_map
    )

    # Determine which channels are present
    labels = []
    if red_used: labels.append('Red (%)')
    if green_used: labels.append('Green (%)')
    if blue_used: labels.append('Blue (%)')

    # for (cmap, norm, lbl) in zip(cmaps, norms, labels):
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #     sm.set_array([])
    #     cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    #     cbar.set_label(lbl)

    plt.title('Duct System Colored by Channel Percentages')
    plt.show()


if __name__ == "__main__":
    # json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\890_annotations.json'
    # duct_borders_path =  r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max.lif - TileScan 1 Merged_Processed001_duct.geojson'
    #
    #
    # # Set channels (use None if not available)
    # red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max_forbranchanalysis-0003.tif'
    # blue_image_path = None
    # green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max_forbranchanalysis-0004.tif'

    # json_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'
    # duct_borders_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\annotations_exported.geojson'
    #
    # red_image_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\28052024_2435322_L5_ecad_mAX-0006.tif'
    # blue_image_path = None
    # green_image_path = None
    # threshold_value = 500

    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

    blue_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
    green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
    red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
    threshold_value = 200

    duct_systems = load_duct_systems(json_path)
    system_idx = 1
    duct_system = duct_systems[system_idx]

    if len(duct_system["segments"]) > 0:
        duct_system = clean_duct_data(duct_system)
        main_branch_node = list(duct_system["branch_points"].keys())[0]
        duct_system = simplify_duct_system(duct_system, main_branch_node)

    G = create_duct_graph(duct_system)

    seg_red, tot_red, pos_red = compute_segment_percentages(red_image_path, duct_system, duct_borders_path, threshold_value)
    seg_green, tot_green, pos_green = compute_segment_percentages(green_image_path, duct_system, duct_borders_path, threshold_value)
    seg_blue, tot_blue, pos_blue = compute_segment_percentages(blue_image_path, duct_system, duct_borders_path, threshold_value)

    # Prepare list of dicts (omit empty)
    percentages_list = [d for d in [seg_red, seg_green, seg_blue] if d]

    segment_color_map, cmaps, norms = create_segment_color_map(duct_system, percentages_list)

    # Filter cmaps/norms based on which channels are used
    channel_used = [red_image_path is not None, green_image_path is not None, blue_image_path is not None]
    cmaps_final = [c for c, u in zip(cmaps, channel_used) if u]
    norms_final = [n for n, u in zip(norms, channel_used) if u]

    # Print overall percentages
    if tot_red > 0:
        print(f"Overall Red: {pos_red}/{tot_red} = {pos_red/tot_red*100:.2f}%")
    if tot_green > 0:
        print(f"Overall Green: {pos_green}/{tot_green} = {pos_green/tot_green*100:.2f}%")
    if tot_blue > 0:
        print(f"Overall Blue: {pos_blue}/{tot_blue} = {pos_blue/tot_blue*100:.2f}%")

    plot_with_channels(duct_system, G, segment_color_map, cmaps_final, norms_final,
                       red_used=(red_image_path is not None),
                       green_used=(green_image_path is not None),
                       blue_used=(blue_image_path is not None))
