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

from utils.fixing_annotations import simplify_graph
from utils.plotting_trees import plot_hierarchical_graph
from utils.loading_saving import load_duct_systems, create_directed_duct_graph


def compute_segment_percentages_from_graph(image_path, G, borders_path, threshold):
    """
    Computes per-segment positive pixel percentages from an image,
    using the duct geometry stored in the graph G.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    G : nx.Graph
        Graph with branch point and segment metadata.
    borders_path : str
        Path to a GeoJSON file of duct borders.
    threshold : numeric
        Pixel intensity threshold for considering a pixel positive.

    Returns
    -------
    seg_percentages : dict
        Dictionary mapping each segment name to its positive pixel percentage.
    overall_total : int
        Total number of pixels within segments.
    overall_positive : int
        Total number of positive pixels within segments.
    """
    if image_path is None:
        return {}, 0, 0

    image = io.imread(image_path)
    with open(borders_path, 'r') as f:
        duct_borders = json.load(f)

    shapes_list = [(shape(feat['geometry']), 1) for feat in duct_borders['features']]
    mask = rasterize(shapes_list, out_shape=image.shape, fill=0, dtype=np.uint8, all_touched=False)

    masked_image = image.copy()
    masked_image[mask == 0] = 0
    binary_image = masked_image > threshold

    segment_shapes = []
    segment_id_mapping = {}
    # Iterate over all edges in the graph.
    for idx, (u, v, data) in enumerate(G.edges(data=True), start=1):
        seg_name = data.get('segment_name')
        segment_id_mapping[idx] = seg_name
        start_bp = G.nodes[u]
        end_bp = G.nodes[v]
        pts = [(start_bp['x'], start_bp['y'])]
        if data.get('internal_points'):
            pts.extend([(p['x'], p['y']) for p in data.get('internal_points', [])])
        pts.append((end_bp['x'], end_bp['y']))
        # Buffer the line by 5 pixels.
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

    overall_total = np.sum(total[1:])  # skip index 0
    overall_positive = np.sum(positive[1:])
    return seg_percentages, overall_total, overall_positive


def create_segment_color_map_from_percentages(percentages_list):
    """
    Generates a segment color map based on percentage values from multiple channels.
    Expects percentages_list to be a list of dictionaries, each mapping segment names to percentage values.
    Uses three channels: red, green, and yellow.

    Returns:
      segment_color_map: dict mapping segment name to HEX color.
      cmaps: list of matplotlib colormaps.
      norms: list of normalization objects.
    """
    # Set channel order to red, green, yellow.
    num_channels = len(percentages_list)
    channels = ['red', 'green', 'yellow']
    max_percentages = [max(d.values()) if d else 1 for d in percentages_list]

    cmaps, norms = [], []
    for i in range(num_channels):
        cmap = LinearSegmentedColormap.from_list(f"my_{channels[i]}", [(0, 'black'), (1, channels[i])])
        cmaps.append(cmap)
        norms.append(colors.Normalize(vmin=0, vmax=max_percentages[i]))

    # Use keys from the first dictionary as segment names.
    segments = percentages_list[0].keys() if percentages_list else []
    segment_color_map = {}
    for seg_name in segments:
        vals = [d.get(seg_name, 0) for d in percentages_list]
        rgb = [int((v / 10) * 255) for v in vals]
        # Apply non-linear scaling (optional).
        rgb = [int(255 * (v / 255) ** 0.3) for v in rgb]
        while len(rgb) < 3:
            rgb.append(0)
        segment_color_map[seg_name] = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    return segment_color_map, cmaps, norms


def plot_with_channels(G, segment_color_map, cmaps, norms, red_used, green_used, yellow_used):
    """
    Plots the duct system using a hierarchical graph function, coloring segments
    according to the provided segment_color_map. Relies solely on graph G.
    """
    fig, ax = plot_hierarchical_graph(
        G,
        root_node=list(G.nodes)[0],
        use_hierarchy_pos=True,
        vert_gap=8,
        linewidth=1.1,
        orthogonal_edges=True,
        annotation_to_color={},  # not used here
        segment_color_map=segment_color_map
    )
    plt.title('Duct System Colored by Channel Percentages')
    plt.show()
    return fig, ax


if __name__ == "__main__":
    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

    green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
    yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
    red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
    threshold_value = 400
    system_idx = 1

    duct_systems = load_duct_systems(json_path)
    duct_system = duct_systems[system_idx]

    # Build and simplify the graph.
    G = create_directed_duct_graph(duct_system)
    G = simplify_graph(G)

    # Compute segment percentages for each channel.
    seg_red, tot_red, pos_red = compute_segment_percentages_from_graph(red_image_path, G, duct_borders_path,
                                                                       threshold_value)
    seg_green, tot_green, pos_green = compute_segment_percentages_from_graph(green_image_path, G, duct_borders_path,
                                                                             threshold_value)
    seg_yellow, tot_yellow, pos_yellow = compute_segment_percentages_from_graph(yellow_image_path, G, duct_borders_path,
                                                                                threshold_value)

    # Prepare percentages list in the order: red, green, yellow.
    percentages_list = [d for d in [seg_red, seg_green, seg_yellow] if d]
    segment_color_map, cmaps, norms = create_segment_color_map_from_percentages(percentages_list)

    # Determine which channels are used.
    channel_used = [red_image_path is not None, green_image_path is not None, yellow_image_path is not None]
    cmaps_final = [c for c, u in zip(cmaps, channel_used) if u]
    norms_final = [n for n, u in zip(norms, channel_used) if u]

    if tot_red > 0:
        print(f"Overall Red: {pos_red}/{tot_red} = {pos_red / tot_red * 100:.2f}%")
    if tot_green > 0:
        print(f"Overall Green: {pos_green}/{tot_green} = {pos_green / tot_green * 100:.2f}%")
    if tot_yellow > 0:
        print(f"Overall Yellow: {pos_yellow}/{tot_yellow} = {pos_yellow / tot_yellow * 100:.2f}%")

    plot_with_channels(G, segment_color_map, cmaps_final, norms_final,
                       red_used=(red_image_path is not None),
                       green_used=(green_image_path is not None),
                       yellow_used=(yellow_image_path is not None))
