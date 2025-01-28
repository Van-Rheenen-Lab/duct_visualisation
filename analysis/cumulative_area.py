import networkx as nx
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from shapely.geometry import LineString, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from rasterio.features import rasterize
import json


def compute_bfs_levels(G_dir, root_node):
    """
    Returns a dict: {node: level}, BFS distance from `root_node`.
    Assumes G_dir is a DiGraph.
    """
    if root_node not in G_dir:
        raise ValueError(f"Root node {root_node} not in the graph.")
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
    Returns a dict {level: [segment_names]}
    Each edge is grouped by BFS level of its end node.
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
    Return a LineString for the given segment.
    Example logic that assumes:
      duct_system['segments'][segment_name] = {
        'start_bp': <bp_key>, 'end_bp': <bp_key>,
        'internal_points': [ {x:..., y:...}, ... ]
      }
      duct_system['branch_points'][bp_key] = { x:..., y:... }
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

########################################################
# 2) Corridor-based Pixel Extraction
########################################################
def pixels_for_segment_multi_corridor(
    segment_line,
    duct_polygon,
    images,
    threshold=1000,
    buffer_dist=5.0,
    out_shape=None
):
    """
    Build a corridor by buffering segment_line in real coordinates.
    Intersect with duct_polygon, rasterize the corridor.
    Return:
      all_pixels: set of (row, col)
      channel_positive: {channel_idx -> set of (row, col)} above threshold.

    If everything is empty, returns empty sets.
    """
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
        all_touched=True  # or False, as you prefer
    )
    ys, xs = np.where(corridor_mask == 1)
    if len(xs) == 0:
        return set(), {i: set() for i in range(len(images))}

    all_pixels = set(zip(ys, xs))

    # 4) Check each channel’s intensity for threshold
    channel_positive = {}
    for i, img in enumerate(images):
        if img is None:
            channel_positive[i] = set()
            continue
        pos_pix = set()
        for (row, col) in all_pixels:
            if img[row, col] > threshold:
                pos_pix.add((row, col))
        channel_positive[i] = pos_pix

    return all_pixels, channel_positive

########################################################
# 3) Multi-channel BFS with corridor-based pixels
########################################################
from shapely.ops import unary_union
from shapely.geometry import LineString
import numpy as np
import rasterio.features

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import rasterio.features


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
    Optimized BFS corridor analysis that:
      - Groups segments by BFS level.
      - Unions all corridor polygons at BFS level i => single geometry.
      - Maintains a cumulative corridor polygon (P_i) as we go from 1..max_level.
      - For each new pixel discovered at BFS level i, sets pixel_levels[y,x] = i.

    Returns:
      area_by_level: list of cumulative duct area up to BFS level i, for i=0..max_level
      positives_by_level: {channel_idx -> [cumulative positives], 0..max_level}
      pixel_levels: 2D np.uint16 array, same shape as images, where:
         pixel_levels[y, x] = BFS level that first included that pixel
         0 => not included by any BFS level
    """
    # 1) Basic BFS
    levels = compute_bfs_levels(G_dir, root_node)
    if not levels:
        return [], {}, None

    max_level = max(levels.values())
    level_dict = segments_by_level(G_dir, levels)

    # 2) Output structures
    area_by_level = [0]  # BFS level 0 => 0 area
    num_channels = len(images)
    positives_by_level = {ch: [0] for ch in range(num_channels)}

    # We'll create a BFS-level map for all pixels
    #   shape = out_shape from the first non-None image
    out_shape = None
    for img in images:
        if img is not None:
            out_shape = img.shape
            break
    if out_shape is None:
        raise ValueError("No valid image for out_shape.")

    pixel_levels = np.zeros(out_shape, dtype=np.uint16)

    # 3) Maintain a cumulative corridor polygon across levels
    cumulative_polygon = Polygon()  # empty
    # Also track sets for BFS-level accumulation
    cumulative_pixels = set()
    channel_cumulative = {ch: set() for ch in range(num_channels)}

    # A helper to rasterize a polygon once
    def rasterize_polygon(poly):
        return rasterio.features.rasterize(
            [(poly, 1)],
            out_shape=out_shape,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )

    for lvl in range(1, max_level + 1):
        seg_names = level_dict.get(lvl, [])
        if not seg_names:
            # No segments => no new area, just duplicate
            area_by_level.append(len(cumulative_pixels))
            for ch in range(num_channels):
                positives_by_level[ch].append(len(channel_cumulative[ch]))
            continue

        # 3a) Build corridor union for BFS level i
        corridors_this_level = []
        for seg_name in seg_names:
            line = get_line_for_segment(duct_system, seg_name)
            buff = line.buffer(buffer_dist)
            corridor_polygon = buff.intersection(duct_polygon)
            if not corridor_polygon.is_empty:
                corridors_this_level.append(corridor_polygon)

        if corridors_this_level:
            level_union_polygon = unary_union(corridors_this_level)
        else:
            level_union_polygon = Polygon()  # empty

        # 3b) Union with existing cumulative corridor
        new_cumulative = unary_union([cumulative_polygon, level_union_polygon])
        if new_cumulative.equals(cumulative_polygon):
            # No change
            area_by_level.append(len(cumulative_pixels))
            for ch in range(num_channels):
                positives_by_level[ch].append(len(channel_cumulative[ch]))
            continue

        # 3c) Instead of re-rasterizing both polygons, compute the difference polygon
        #     to identify "newly added" region. Then we can rasterize that difference.
        difference_polygon = new_cumulative.difference(cumulative_polygon)
        # Rasterize only newly added area
        diff_mask = rasterize_polygon(difference_polygon)

        # new pixels
        ys, xs = np.where(diff_mask == 1)
        new_pix = set(zip(ys, xs))

        # Assign BFS level to these new pixels in pixel_levels,
        # but only if pixel_levels[y,x] == 0 (i.e., they were not set yet).
        for (row, col) in new_pix:
            if pixel_levels[row, col] == 0:
                pixel_levels[row, col] = lvl

        # Update the global sets
        cumulative_pixels.update(new_pix)
        for ch in range(num_channels):
            if images[ch] is None:
                continue
            img = images[ch]
            above_thresh = []
            for (row, col) in new_pix:
                if img[row, col] > threshold:
                    above_thresh.append((row, col))
            channel_cumulative[ch].update(above_thresh)

        # Update the polygon
        cumulative_polygon = new_cumulative

        # 3d) Record BFS i
        area_by_level.append(len(cumulative_pixels))
        for ch in range(num_channels):
            positives_by_level[ch].append(len(channel_cumulative[ch]))

    return area_by_level, positives_by_level, pixel_levels

import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_pixel_levels(pixel_levels, max_level):
    """
    pixel_levels: 2D array of BFS levels (0..max_level)
    any 0 => not assigned
    """
    # Build a discrete set of colors:
    # level 0 => black, level 1..max_level => some colormap slices
    color_list = ["black"]
    cmap_viridis = plt.cm.get_cmap("viridis", max_level)  # or any named colormap
    for i in range(max_level):
        # i goes 0..(max_level-1); shift by +1 to pick BFS level color
        color_list.append(mpl.colors.rgb2hex(cmap_viridis(i)))

    discrete_cmap = mpl.colors.ListedColormap(color_list)

    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.imshow(pixel_levels, cmap=discrete_cmap, vmin=0, vmax=max_level)
    ax.set_title("Pixel BFS Level Map")
    ax.axis("off")

    # Colorbar
    ticks = range(max_level+1)  # 0..max_level
    cbar = plt.colorbar(cax, ticks=ticks)
    cbar.ax.set_yticklabels([str(i) for i in ticks])
    cbar.set_label("BFS Level")
    plt.tight_layout()
    plt.show()


########################################################
# 4) Stack Plot for Multi-channel BFS
########################################################
def plot_area_vs_branch_level_multi_stack(area_by_level, positives_by_level,
                                          channel_labels=None, use_log=False):
    """
    Produce a stack plot of BFS-level-based cumulative area, where:
      - Row 0 = Negative area
      - Row i = channel i's positives (stacked above)
    Overlapping pixels in multiple channels are double-counted in the stack.

    area_by_level: list of total duct area (index = BFS level, 0..N)
    positives_by_level: {channel_idx -> [cumulative positives], length matches area_by_level}
    """
    if channel_labels is None:
        channel_labels = [f"Channel {i}" for i in sorted(positives_by_level.keys())]

    xvals = range(len(area_by_level))  # BFS levels from 0..(len-1)

    # Sum each channel's positives
    sum_of_positives = []
    sorted_channels = sorted(positives_by_level.keys())

    for lvl in xvals:
        total_pos = 0
        for ci in sorted_channels:
            total_pos += positives_by_level[ci][lvl]
        sum_of_positives.append(total_pos)

    # Negative portion = total duct area - sum_of_positives
    negative = []
    for lvl_idx, total_area in enumerate(area_by_level):
        val = total_area - sum_of_positives[lvl_idx]
        if val < 0:
            val = 0  # clamp to zero if overlap is greater than total
        negative.append(val)

    # Build data series for stackplot
    stacked_data = [negative]
    for ci in sorted_channels:
        stacked_data.append(positives_by_level[ci])

    labels = ["Negative"] + [channel_labels[ci] for ci in sorted_channels]

    # pick some colors
    colors = ["#000000", "red", "green", "yellow"]

    fig, ax = plt.subplots(figsize=(8, 6))
    if use_log:
        ax.set_yscale('log')

    ax.stackplot(xvals, *stacked_data, labels=labels, colors=colors[:len(labels)])
    ax.set_xlabel("Branch Level")
    ax.set_ylabel("Cumulative Area (pixels)")
    ax.set_title("Stacked Cumulative Area vs. Branch Level")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from utils.loading_saving import load_duct_systems, create_directed_duct_graph
    from skimage import io

    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

    green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
    yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
    red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'

    threshold_value = 500
    system_idx = 1

    # json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\890_annotations.json'
    # duct_borders_path =  r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max.lif - TileScan 1 Merged_Processed001_duct.geojson'
    #
    # red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max_forbranchanalysis-0003.tif'
    # yellow_image_path = None
    # green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2475890_BhomPhet_24W\05112024_2475890_L4_sma_mp1_max_forbranchanalysis-0001.tif'
    # system_idx = 2
    # threshold_value = 1000
    #
    # json_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2437324_BhomPhom_24W\annotations 7324-1.json"
    # duct_borders_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2437324_BhomPhom_24W\01062024_7324_L5_sma_max.lif - TileScan 1 Merged_Processed001_forbranchanalysis.geojson"
    #
    # red_image_path = r"I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2437324_BhomPhom_24W\30052024_7324_L5_sma_max_clean_forbranchanalysis-0005.tif"
    # green_image_path = None
    # yellow_image_path = None
    # system_idx = 1
    # threshold_value = 1000



    # Load images if they exist, else None
    def safe_read(path):
        return io.imread(path) if path else None

    red_image = safe_read(red_image_path)
    green_image = safe_read(green_image_path)
    yellow_image = safe_read(yellow_image_path)

    # Put them in a list
    images = [red_image, green_image, yellow_image]
    channel_labels = ["Red channel", "Green channel", "Yellow channel"]

    # Load the duct system
    duct_systems = load_duct_systems(json_path)
    duct_system = duct_systems[system_idx]

    # Build directed graph
    G_dir = create_directed_duct_graph(duct_system)

    # Find a BFS root (example logic)
    first_bp = list(G_dir.nodes)[0]
    while len(list(G_dir.predecessors(first_bp))) == 1:
        first_bp = list(G_dir.predecessors(first_bp))[0]
    root_node = first_bp
    print(f"Starting from first branch point: {root_node}")

    # Build duct polygon mask
    with open(duct_borders_path, 'r') as f:
        duct_borders = json.load(f)

    valid_geoms = []
    for feat in duct_borders['features']:
        geom = shape(feat['geometry'])
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_valid:
            valid_geoms.append(geom)
        else:
            print("Skipping geometry that could not be fixed")

    duct_polygon = unary_union(valid_geoms)

    # Use one of the non-None images to get shape, else fallback
    base_shape = None
    for img in images:
        if img is not None:
            base_shape = img.shape
            break

    if base_shape is None:
        raise ValueError("No valid image to determine shape.")

    shapes = [(duct_polygon, 1)]
    duct_mask = rasterize(shapes, out_shape=base_shape, fill=0, dtype=np.uint8, all_touched=False)


    area_by_lvl, positives_by_lvl, pixel_levels = analyze_area_vs_branch_level_multi_corridor(
        duct_system=duct_system,
        G_dir=G_dir,
        root_node=root_node,
        duct_polygon=duct_polygon,
        images=images,
        threshold=500,
        buffer_dist=5.0
    )

    # Then do your stack plot
    plot_area_vs_branch_level_multi_stack(
        area_by_lvl, positives_by_lvl,
        channel_labels=["Red", "Green", "Yellow"],
        use_log=False
    )

    # Finally, color-code BFS levels
    max_level = len(area_by_lvl) - 1
    plot_pixel_levels(pixel_levels, max_level)