import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

from skimage import io
from collections import deque
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from rasterio.features import rasterize


###############################################################################
# 1) Basic Graph + BFS Helpers
###############################################################################
def compute_bfs_levels(G_dir, root_node):
    """
    Returns a dict {node: level} indicating the BFS distance from `root_node`.
    Only follows directed edges in the forward ("successors") direction.
    """
    if root_node not in G_dir:
        raise ValueError(f"Root node {root_node} is not in the graph.")
    levels = {}
    queue = deque([(root_node, 0)])
    visited = set()

    while queue:
        current, dist = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        levels[current] = dist
        # Only move downstream
        for child in G_dir.successors(current):
            if child not in visited:
                queue.append((child, dist + 1))

    return levels


def segments_by_level(G_dir, levels):
    """
    Returns a dict {level: [segment_names]}.
    An edge from u->v is grouped by the BFS level of v.
    """
    level_dict = {}
    for u, v in G_dir.edges():
        seg_name = G_dir[u][v].get('segment_name', None)
        if seg_name is None:
            continue
        end_level = levels[v]
        level_dict.setdefault(end_level, []).append(seg_name)
    return level_dict


def get_line_for_segment(duct_system, segment_name):
    """
    Given a duct_system and a segment_name, return the corresponding LineString.
    Assumes duct_system['segments'][segment_name] has 'start_bp', 'end_bp',
    plus optionally 'internal_points'.
    """
    segments = duct_system['segments']
    branch_points = duct_system['branch_points']

    seg_data = segments[segment_name]
    start_bp = branch_points[seg_data['start_bp']]
    end_bp = branch_points[seg_data['end_bp']]

    pts = [(start_bp['x'], start_bp['y'])]
    pts.extend([(p['x'], p['y']) for p in seg_data.get('internal_points', [])])
    pts.append((end_bp['x'], end_bp['y']))

    return LineString(pts)


###############################################################################
# 2) Corridor-based Pixel Extraction (Optional standalone function)
###############################################################################
def pixels_for_segment_multi_corridor(
    segment_line,
    duct_polygon,
    images,
    threshold=1000,
    buffer_dist=5.0,
    out_shape=None
):
    """
    Build a corridor by buffering `segment_line`. Intersect with `duct_polygon`,
    rasterize the result, and check intensities in each channel against threshold.

    Returns:
      all_pixels: set of (row, col)
      channel_positive: {channel_idx -> set of (row, col)}
    """
    if out_shape is None or len(images) == 0:
        return set(), {}

    # 1) Buffer the line
    corridor_polygon = segment_line.buffer(buffer_dist)

    # 2) Intersect with duct
    corridor_polygon = corridor_polygon.intersection(duct_polygon)
    if corridor_polygon.is_empty:
        return set(), {i: set() for i in range(len(images))}

    # 3) Rasterize corridor
    corridor_mask = rasterize(
        [(corridor_polygon, 1)],
        out_shape=out_shape,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    ys, xs = np.where(corridor_mask == 1)
    if len(xs) == 0:
        return set(), {i: set() for i in range(len(images))}

    all_pixels = set(zip(ys, xs))

    # 4) Check intensities for threshold
    channel_positive = {}
    for i, img in enumerate(images):
        if img is None:
            channel_positive[i] = set()
            continue
        pos_pix = {(row, col) for (row, col) in all_pixels if img[row, col] > threshold}
        channel_positive[i] = pos_pix

    return all_pixels, channel_positive


###############################################################################
# 3) Multi-channel BFS with corridor-based pixels (Optimized version)
###############################################################################
def analyze_area_vs_branch_level_multi_corridor(
    duct_system,
    G_dir,
    root_node,
    duct_polygon,
    images,
    threshold=1000,
    buffer_dist=5.0
):
    """
    Perform BFS-level corridor analysis:
      1) Compute BFS levels from `root_node`.
      2) Group edges/segments by BFS level.
      3) For level i:
         - Union all corridor polygons for segments at level i,
           intersect with duct_polygon.
         - Rasterize to get corridor_mask_i.
         - new_pixels = corridor_mask_i & ~cumulative_mask (i.e. newly added).
         - Mark those new pixels in `pixel_levels` with BFS level i.
         - Update cumulative coverage in raster domain via OR operation.
         - Count how many new pixels pass threshold in each channel.

    Returns:
      area_by_level: [cumulative area at BFS level i] for i=0..max_level
      positives_by_level: {channel_idx: [cumulative positives at each BFS level]}
      pixel_levels: 2D array with BFS level per pixel (0 => not included)
    """
    # 1) BFS to get levels
    levels = compute_bfs_levels(G_dir, root_node)
    if not levels:
        return [], {}, None

    max_level = max(levels.values())
    lvl_segments = segments_by_level(G_dir, levels)

    # 2) Determine output shape
    out_shape = None
    for img in images:
        if img is not None:
            out_shape = img.shape
            break
    if out_shape is None:
        raise ValueError("No valid image to determine out_shape.")

    # 3) Prepare outputs
    area_by_level = [0]  # BFS level 0 => 0 area
    num_channels = len(images)
    positives_by_level = {ch: [0] for ch in range(num_channels)}

    # pixel_levels: store the BFS level that first includes a pixel
    pixel_levels = np.zeros(out_shape, dtype=np.uint16)

    # Keep a cumulative boolean mask for all included pixels so far
    cumulative_mask = np.zeros(out_shape, dtype=bool)

    # Keep track of how many positives so far in each channel
    channel_cumulative_count = [0] * num_channels

    def rasterize_polygon(poly):
        # Helper for polygon -> mask
        return rasterize(
            [(poly, 1)],
            out_shape=out_shape,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        ).astype(bool)

    # 4) Iterate BFS levels in ascending order
    for lvl in range(1, max_level + 1):
        seg_names = lvl_segments.get(lvl, [])
        if not seg_names:
            # Nothing at this level => same as previous counts
            area_by_level.append(area_by_level[-1])
            for ch in range(num_channels):
                positives_by_level[ch].append(positives_by_level[ch][-1])
            continue

        # 4a) Union corridor polygons for this BFS level
        corridors = []
        for seg in seg_names:
            line = get_line_for_segment(duct_system, seg)
            buff_poly = line.buffer(buffer_dist)
            corr_poly = buff_poly.intersection(duct_polygon)
            if not corr_poly.is_empty:
                corridors.append(corr_poly)

        if corridors:
            union_poly = unary_union(corridors)
        else:
            union_poly = Polygon()  # empty

        # 4b) Rasterize BFS level's union corridor
        if union_poly.is_empty:
            # No new area => same as previous
            area_by_level.append(area_by_level[-1])
            for ch in range(num_channels):
                positives_by_level[ch].append(positives_by_level[ch][-1])
            continue

        corridor_mask_i = rasterize_polygon(union_poly)

        # 4c) Compute newly added coverage: corridor_mask_i AND NOT cumulative_mask
        new_mask = corridor_mask_i & (~cumulative_mask)
        new_pixels_indices = np.where(new_mask)

        # Assign BFS level to these new pixels
        pixel_levels[new_pixels_indices] = lvl

        # Update the cumulative coverage in the raster domain
        cumulative_mask |= corridor_mask_i

        # 4d) Count how many new pixels pass the threshold in each channel
        new_pixels_count = np.count_nonzero(new_mask)
        current_area_total = area_by_level[-1] + new_pixels_count

        # For each channel, see how many newly added pass threshold
        for ch in range(num_channels):
            if images[ch] is not None:
                ch_image = images[ch]
                # Boolean mask for "above threshold" among newly added
                above_thr_mask = (ch_image > threshold) & new_mask
                positives_count_new = np.count_nonzero(above_thr_mask)
                channel_cumulative_count[ch] += positives_count_new
            # Append the updated channel total for BFS level
            positives_by_level[ch].append(channel_cumulative_count[ch])

        # Finally, record the total duct coverage so far at BFS level
        area_by_level.append(current_area_total)

    return area_by_level, positives_by_level, pixel_levels


###############################################################################
# 5) Visualization Helpers
###############################################################################
def plot_pixel_levels(pixel_levels, max_level):
    """
    Visualize BFS levels assigned to each pixel:
      0 => not included
      1..max_level => BFS corridor levels
    """
    # Build a discrete colormap: 0 => black, 1..max_level => viridis
    color_list = ["black"]
    cmap_viridis = plt.cm.get_cmap("viridis", max_level)
    for i in range(max_level):
        color_list.append(mpl.colors.rgb2hex(cmap_viridis(i)))
    discrete_cmap = mpl.colors.ListedColormap(color_list)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(pixel_levels, cmap=discrete_cmap, vmin=0, vmax=max_level)
    ax.set_title("Pixel Branch Level Map")
    ax.axis("off")

    # Colorbar
    ticks = range(max_level + 1)  # 0..max_level
    cbar = plt.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) for i in ticks])
    cbar.set_label("Branch Level")
    plt.tight_layout()



def plot_area_vs_branch_level_multi_stack(
    area_by_level,
    positives_by_level,
    channel_labels=None,
    use_log=False
):
    """
    Produce a stack plot showing how the cumulative duct area (and each channel’s
    positive pixels) grow over BFS levels. Overlapping positives in multiple
    channels are double-counted in the stack.
    """
    if channel_labels is None:
        channel_labels = [f"Channel {i}" for i in sorted(positives_by_level.keys())]

    xvals = range(len(area_by_level))  # BFS levels from 0..(len-1)
    sorted_channels = sorted(positives_by_level.keys())

    # Sum each channel's positives
    sum_of_positives = []
    for lvl in xvals:
        total_pos = sum(positives_by_level[ch][lvl] for ch in sorted_channels)
        sum_of_positives.append(total_pos)

    # Negative portion = total duct area - sum_of_positives
    negative = []
    for i, total_area in enumerate(area_by_level):
        val = total_area - sum_of_positives[i]
        negative.append(val if val > 0 else 0)

    # Build data series for stackplot
    stacked_data = []
    for ch in sorted_channels:
        stacked_data.append(positives_by_level[ch])

    stacked_data.append(negative)

    labels = [channel_labels[ch] for ch in sorted_channels] + ["Negative"]

    colors = ["red", "green", "yellow", "#000000"]

    fig, ax = plt.subplots(figsize=(8, 6))
    if use_log:
        ax.set_yscale('log')

    ax.stackplot(
        xvals,
        *stacked_data,
        labels=labels,
        colors=colors[:len(labels)]
    )
    ax.set_xlabel("Branch Level")
    ax.set_ylabel("Cumulative Area (pixels)")
    ax.set_title("Stacked Cumulative Area vs. Branch Level")
    ax.legend(loc="best")
    plt.tight_layout()



###############################################################################
# 6) Example Main
###############################################################################
if __name__ == "__main__":
    from shapely.validation import make_valid


    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

    green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
    yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
    red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
    threshold_value = 1000
    system_idx = 1

    ## BhomPhet
    # json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\890_annotations.json'
    # duct_borders_path =  r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max.lif - TileScan 1 Merged_Processed001_duct.geojson'
    #
    # red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max_forbranchanalysis-0003.tif'
    # yellow_image_path = None
    # green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max_forbranchanalysis-0001.tif'
    # system_idx = 2
    # threshold_value = 1000
    #
    # # Phet
    # red_image_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\28052024_2435322_L5_ecad_mAX-0006.tif'
    # duct_borders_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\annotations_exported.geojson'
    # json_path = r'I:\Group Rheenen\ExpDATA\2024_J.DOORNBOS\004_ToolDev_duct_annotation_tool\Duct annotations example hris\normalized_annotations.json'
    # green_image_path = None
    # yellow_image_path = None
    # threshold_value = 1000
    # system_idx = 0

    ## BhomPhom
    # json_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2437324_BhomPhom_24W\annotations 7324-1.json"
    # duct_borders_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2437324_BhomPhom_24W\01062024_7324_L5_sma_max.lif - TileScan 1 Merged_Processed001_forbranchanalysis.geojson"
    #
    # red_image_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2437324_BhomPhom_24W\30052024_7324_L5_sma_max_clean_forbranchanalysis-0005.tif"
    # green_image_path = None
    # yellow_image_path = None
    # system_idx = 1
    # threshold_value = 1000



    # Safe loading
    def safe_read(path):
        return io.imread(path) if path else None

    red_image = safe_read(red_image_path)
    green_image = safe_read(green_image_path)
    yellow_image = safe_read(yellow_image_path)
    images = [red_image, green_image, yellow_image]
    channel_labels = ["Red", "Green", "Yellow"]

    # -------------------------------------------------------
    # Load the duct systems from JSON (stub function)
    # Replace with your real function that returns a list of duct systems
    def load_duct_systems(json_file):
        # E.g., you have a function that returns list of systems
        with open(json_file, 'r') as f:
            all_systems = json.load(f)
        return all_systems["duct_systems"]

    # Example: create a directed graph from the duct system
    def create_directed_duct_graph(duct_system):
        G = nx.DiGraph()
        for seg_name, seg_data in duct_system['segments'].items():
            start_bp = seg_data['start_bp']
            end_bp = seg_data['end_bp']
            # Add edge, store segment_name
            G.add_edge(start_bp, end_bp, segment_name=seg_name)
        return G

    # -------------------------------------------------------
    # Load data
    duct_systems = load_duct_systems(json_path)
    duct_system = duct_systems[system_idx]

    G_dir = create_directed_duct_graph(duct_system)

    # Find a BFS root (toy example: just climb upstream while there's exactly 1 predecessor)
    first_bp = list(G_dir.nodes)[0]
    while len(list(G_dir.predecessors(first_bp))) == 1:
        first_bp = list(G_dir.predecessors(first_bp))[0]
    root_node = first_bp
    print(f"Using branch point {root_node} as BFS root")

    # Load duct borders from geojson, fix invalid geometry
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

    # -------------------------------------------------------
    area_by_lvl, positives_by_lvl, pixel_levels = analyze_area_vs_branch_level_multi_corridor(
        duct_system=duct_system,
        G_dir=G_dir,
        root_node=root_node,
        duct_polygon=duct_polygon,
        images=images,
        threshold=threshold_value,
        buffer_dist=10.0
    )

    # Plot
    plot_area_vs_branch_level_multi_stack(
        area_by_lvl,
        positives_by_lvl,
        channel_labels=channel_labels,
        use_log=False
    )

    # Show BFS level assignment
    max_level = len(area_by_lvl) - 1
    plot_pixel_levels(pixel_levels, max_level)
    plt.show()
    print("Analysis complete!")
