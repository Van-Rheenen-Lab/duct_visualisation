import networkx as nx
from collections import deque
import numpy as np
import warnings
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from shapely.geometry import shape
from rasterio.features import rasterize
from shapely.validation import make_valid
from shapely.ops import unary_union
import json

########################################################
# Original BFS code remains the same
########################################################

def compute_bfs_levels(G_dir, root_node):
    """
    Returns a dict: {node: level}, where `level` is the BFS distance from `root_node`.
    Assumes G_dir is a DiGraph and there's exactly one path from root to each node.
    """
    if root_node not in G_dir:
        raise ValueError(f"Root node {root_node} is not in the graph.")

    from collections import deque
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
    Returns a dict: {level: [segment_names]}.
    Each segment is classified by the BFS level of its end node.
    """
    level_dict = {}
    for u, v in G_dir.edges():
        seg_name = G_dir[u][v].get('segment_name', None)
        if seg_name is None:
            continue
        end_level = levels[v]  # BFS level of the child node
        level_dict.setdefault(end_level, []).append(seg_name)
    return level_dict


def get_line_for_segment(duct_system, segment_name):
    """
    Re-use your existing logic or import from your code.
    Returns a LineString for the specified segment.
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
# Updated pixel / multi-channel code
########################################################
def pixels_for_segment_multi(line, duct_mask, images, threshold=1000, buffer_width=5):
    minx, miny, maxx, maxy = line.bounds
    height, width = duct_mask.shape

    minx = max(int(minx - buffer_width), 0)
    miny = max(int(miny - buffer_width), 0)
    maxx = min(int(maxx + buffer_width), width - 1)
    maxy = min(int(maxy + buffer_width), height - 1)

    sub_mask = duct_mask[miny:maxy+1, minx:maxx+1]
    ys, xs = np.where(sub_mask == 1)
    if len(xs) == 0:
        # Return *empty* sets for *all* channels
        return set(), {i: set() for i in range(len(images))}

    # Shift back to full image coords
    xs += minx
    ys += miny
    all_pixels = set(zip(ys, xs))

    # Build a dictionary for each channel
    channel_positive = {}
    for i, img in enumerate(images):
        if img is None:
            # This channel is missing => empty set
            channel_positive[i] = set()
        else:
            # Filter the pixels that are above threshold
            pos_pix = set()
            for (row, col) in all_pixels:
                if img[row, col] > threshold:
                    pos_pix.add((row, col))
            channel_positive[i] = pos_pix

    return all_pixels, channel_positive

import numpy as np

def get_segment_pixels(line, duct_mask, buffer_width=5):
    """
    Return the set of (y, x) pixels in duct_mask that lie
    in the bounding box of 'line' with a small buffer.
    (No threshold logic; just the polygon mask.)
    """
    minx, miny, maxx, maxy = line.bounds
    height, width = duct_mask.shape

    minx = max(int(minx - buffer_width), 0)
    miny = max(int(miny - buffer_width), 0)
    maxx = min(int(maxx + buffer_width), width - 1)
    maxy = min(int(maxy + buffer_width), height - 1)

    sub_mask = duct_mask[miny:maxy+1, minx:maxx+1]
    ys, xs = np.where(sub_mask == 1)
    if len(xs) == 0:
        return set()

    xs += minx
    ys += miny
    return set(zip(ys, xs))

def compute_pixel_levels(duct_system, G_dir, root_node, duct_mask, buffer_width=5):
    """
    Build an integer array 'pixel_levels' the same shape as duct_mask,
    where pixel_levels[y, x] = BFS level of that pixel (or 0 if not yet assigned).
    """
    # BFS levels => {node: level}
    levels = compute_bfs_levels(G_dir, root_node)
    if not levels:
        raise ValueError("No BFS levels found; check your graph.")

    max_level = max(levels.values())
    # Segments by BFS level => {level: [segment_names]}
    level_dict = segments_by_level(G_dir, levels)

    pixel_levels = np.zeros_like(duct_mask, dtype=np.uint16)

    # For BFS levels 1..max_level, fill in pixel_levels
    for lvl in range(1, max_level + 1):
        seg_names = level_dict.get(lvl, [])
        for seg_name in seg_names:
            line = get_line_for_segment(duct_system, seg_name)
            all_pixels = get_segment_pixels(line, duct_mask, buffer_width)
            # Label these pixels if not already labeled
            for (row, col) in all_pixels:
                if pixel_levels[row, col] == 0:
                    pixel_levels[row, col] = lvl

    return pixel_levels, max_level

import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_pixel_levels(pixel_levels, max_level):
    """
    pixel_levels: 2D np.array of shape = duct_mask.shape
    max_level: highest BFS level

    Plots a discrete color-coded map of BFS levels.
    """
    # Create a color list for 0..max_level
    # 0 => black, then 1..max_level => a colormap like 'viridis' or any you like.
    color_list = ['black']
    cmap_viridis = plt.cm.get_cmap('viridis', max_level)  # discrete slices
    for i in range(max_level):
        color_list.append(mpl.colors.rgb2hex(cmap_viridis(i)))

    # Or use e.g. plt.cm.tab10 or any multi‐color map for BFS levels
    cmap = mpl.colors.ListedColormap(color_list)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(pixel_levels, cmap=cmap, vmin=0, vmax=max_level)
    ax.set_title("Pixel BFS Level Map")
    ax.axis('off')

    # Colorbar with discrete ticks
    cbar = plt.colorbar(cax, ticks=range(max_level + 1))
    cbar.ax.set_yticklabels([str(i) for i in range(max_level + 1)])
    cbar.set_label("BFS Level")
    plt.tight_layout()
    plt.show()


def analyze_area_vs_branch_level_multi(duct_system, G_dir, root_node, duct_mask,
                                       images, threshold=1000, buffer_width=5):
    """
    For each BFS level i, compute the cumulative duct area and the cumulative above-threshold area
    in each of the provided images.

    images: list of up to 3 images (red, green, yellow), but any can be None.
    threshold: intensity threshold.

    Returns:
      area_by_level: A list of total duct area up to each BFS level i (1-based index)
      positives_by_level: A dict of {channel_idx -> [list of cumulative positive area]}
        Each list is the same length as area_by_level.
    """
    # BFS
    levels = compute_bfs_levels(G_dir, root_node)
    max_level = max(levels.values()) if levels else 0

    # Group segments by BFS level of their end node
    level_dict = segments_by_level(G_dir, levels)

    # We track a global set of duct pixels (cumulative) and a global set for each channel.
    cumulative_pixels = set()
    channel_cumulative = {i: set() for i in range(len(images))}

    # We'll store area for BFS levels 0..max_level in a list
    area_by_level = [0]  # area_by_level[i] for BFS level i, i from 1..max_level
    positives_by_level = {i: [0] for i in range(len(images))}

    for level in range(1, max_level+1):
        seg_names = level_dict.get(level, [])
        for seg_name in seg_names:
            line = get_line_for_segment(duct_system, seg_name)
            all_pix, channel_pos = pixels_for_segment_multi(
                line, duct_mask, images, threshold, buffer_width=buffer_width
            )

            # Update the global sets
            cumulative_pixels.update(all_pix)
            for i in range(len(images)):
                channel_cumulative[i].update(channel_pos[i])

        # Record the cumulative area at this level
        area_by_level.append(len(cumulative_pixels))

        # Record each channel's cumulative positives
        for i in range(len(images)):
            positives_by_level[i].append(len(channel_cumulative[i]))

    return area_by_level, positives_by_level


def plot_area_vs_branch_level_multi_stack(area_by_level, positives_by_level,
                                          channel_labels=None, use_log=False):
    """
    Produce a stack plot by BFS level:
      - "Negative" portion at each level
      - Each channel's positive portion stacked above

    WARNING: This is a naive approach. If pixels are positive in multiple
    channels, they get counted multiple times in the stacked region.
    """
    if channel_labels is None:
        # Default channel labels
        channel_labels = [f"Channel {i}"
                          for i in sorted(positives_by_level.keys())]

    # BFS levels are 0..(len(area_by_level)-1)
    xvals = range(len(area_by_level))

    # 1) Sum each channel’s positives by BFS level
    #    positives_by_level is {channel_idx -> [list of positive area counts]}
    sum_of_positives = []
    for lvl in xvals:
        total_pos = 0
        for chan_idx in positives_by_level:
            total_pos += positives_by_level[chan_idx][lvl]
        sum_of_positives.append(total_pos)

    # 2) Negative portion = total duct area - sum_of_positives
    negative = []
    for lvl, total_area in enumerate(area_by_level):
        val = total_area - sum_of_positives[lvl]
        if val < 0:
            # If overlap pushes sum_of_positives above total area,
            # clamp negative portion to zero just so it doesn't go negative.
            val = 0
        negative.append(val)

    # 3) Build data series for stackplot:
    #    - first row = negative
    #    - subsequent rows = each channel's positives
    #    We'll keep channel order sorted by channel index
    stacked_data = []
    sorted_chan_idxs = sorted(positives_by_level.keys())

    for ci in sorted_chan_idxs:
        stacked_data.append(positives_by_level[ci])

    stacked_data.append(negative)

    # 4) Build labels array
    labels = [channel_labels[i] for i in sorted_chan_idxs] + ["Negative"]

    # 5) Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if use_log:
        ax.set_yscale("log")

    colors = ["red", "green", "yellow"]

    ax.stackplot(xvals, *stacked_data, labels=labels,
                 colors= colors + ["#000000"]
    )
    ax.set_xlabel("Branch Level")
    ax.set_ylabel("Cumulative Area (pixels)")
    ax.set_title("Stacked Cumulative Area vs. Branch Level")
    ax.legend(loc="upper left")
    plt.tight_layout()

    # Also show fraction
    fig2, ax2 = plt.subplots(figsize=(8,6))
    for i, (chan_idx, pos_list) in enumerate(positives_by_level.items()):
        c = colors[i % len(colors)]
        # compute fraction of total for BFS levels > 0
        fraction = []
        for lvl_idx in range(1, len(area_by_level)):
            area_val = area_by_level[lvl_idx]
            pos_val = pos_list[lvl_idx]
            frac = (pos_val / area_val) if area_val > 0 else 0
            fraction.append(frac)
        ax2.plot(range(1, len(area_by_level)), fraction, label=channel_labels[chan_idx], color=c)

    ax2.set_xlabel('Branch Level')
    ax2.set_ylabel('Fraction of total area')
    ax2.set_title('Positive Area Fraction vs. Branch Level')
    ax2.legend()
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

    # Analyze all channels
    area_by_lvl, positives_by_lvl = analyze_area_vs_branch_level_multi(
        duct_system, G_dir, root_node, duct_mask,
        images=images, threshold=threshold_value, buffer_width=5
    )
    pixel_levels, max_level = compute_pixel_levels(
        duct_system, G_dir, root_node, duct_mask, buffer_width=5
    )
    plot_pixel_levels(pixel_levels, max_level)

    # Plot results
    plot_area_vs_branch_level_multi_stack(area_by_lvl, positives_by_lvl, channel_labels, use_log=False)
