import numpy as np
import networkx as nx
import warnings
from collections import deque

from shapely.geometry import LineString
from shapely.ops import unary_union
from rasterio.features import rasterize


##############################################################################
# Assume the following two functions exist in your codebase:
#
#   1) load_duct_systems(json_path)
#   2) create_directed_duct_graph(duct_system)
#
# They are not repeated here for brevity.
##############################################################################

def compute_levels(G_dir, root_node):
    """
    Assigns an integer 'level' (branch depth) to each node in G_dir,
    based on distance from the specified root_node.
    """
    if root_node not in G_dir:
        raise ValueError(f"Root node {root_node} not in graph.")
    levels = {}
    visited = set()
    queue = deque([(root_node, 0)])
    while queue:
        node, dist = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        levels[node] = dist
        for child in G_dir.successors(node):
            if child not in visited:
                queue.append((child, dist + 1))
    return levels


def segments_by_level(G_dir, levels):
    """
    Group edges (segments) by the level of their end node.
    Returns {level: [(u, v, seg_name), ...]}.
    """
    level_dict = {}
    for u, v in G_dir.edges():
        seg_name = G_dir[u][v].get('segment_name')
        if seg_name is None:
            continue
        end_level = levels[v]
        level_dict.setdefault(end_level, []).append((u, v, seg_name))
    return level_dict


def get_line_for_segment(duct_system, segment_name):
    """
    Build a shapely LineString from the segment data (including internal points).
    """
    seg_data = duct_system['segments'][segment_name]
    bpoints = duct_system['branch_points']
    start_bp = bpoints[seg_data['start_bp']]
    end_bp = bpoints[seg_data['end_bp']]
    coords = [(start_bp['x'], start_bp['y'])]
    coords += [(p['x'], p['y']) for p in seg_data.get('internal_points', [])]
    coords.append((end_bp['x'], end_bp['y']))
    return LineString(coords)


def analyze_branch_depth_area(
        duct_system,
        G_dir,
        root_node,
        duct_outline,
        images,
        threshold=500,
        buffer_width=5
):
    """
    Compute cumulative area coverage (and positives) vs. branch depth,
    using line-based corridors intersected with the duct outline.

    Parameters
    ----------
    duct_system : dict
      Contains 'branch_points' and 'segments'.
    G_dir : nx.DiGraph
      Directed graph from create_directed_duct_graph(duct_system).
    root_node : str
      Name of the branch point node to treat as depth=0.
    duct_outline : shapely.geometry.Polygon or MultiPolygon
      The actual duct area polygon. We'll intersect corridors with it.
    images : list of ndarray or None
      Each entry is a 2D image (e.g., red, green, yellow).
      All must be same shape: (H, W).
    threshold : int
      Intensity threshold for positivity.
    buffer_width : float
      Corridor radius around each line.

    Returns
    -------
    coverage_by_level : dict
      {level : {"area": cumulative_area, "positives": [pos_channel0, pos_channel1, ...]}}
    pixel_levels : 2D uint16 array
      pixel_levels[r,c] = the branch depth at which (r,c) is first covered.
      0 => not covered.
    """

    # --- 1) Determine shape from first non-None image
    out_shape = None
    for img in images:
        if img is not None:
            out_shape = img.shape
            break
    if out_shape is None:
        raise ValueError("No valid image shape found in `images`.")

    H, W = out_shape

    # --- 2) Compute levels, group segments by level
    levels = compute_levels(G_dir, root_node)
    lvl_segments = segments_by_level(G_dir, levels)
    max_level = max(levels.values()) if levels else 0

    # --- 3) Prepare coverage tracking
    coverage_mask = np.zeros((H, W), dtype=bool)
    pixel_levels = np.zeros((H, W), dtype=np.uint16)
    coverage_by_level = {}
    n_channels = len(images)

    channel_cumul = [0] * n_channels
    cumulative_area = 0

    # --- 4) Iterate each level in ascending order
    for lvl in range(max_level + 1):
        seg_list = lvl_segments.get(lvl, [])
        if not seg_list:
            # No new segments => same area/pos as before
            coverage_by_level[lvl] = {
                "area": cumulative_area,
                "positives": channel_cumul[:]
            }
            continue

        newly_added_pixels = []

        # For each segment, build corridor polygon & intersect with duct outline
        for (u, v, seg_name) in seg_list:
            line = get_line_for_segment(duct_system, seg_name)
            if line.is_empty:
                continue

            corridor_poly = line.buffer(buffer_width)
            if corridor_poly.is_empty or duct_outline.is_empty:
                continue

            intersection_poly = corridor_poly.intersection(duct_outline)
            if intersection_poly.is_empty:
                continue

            # Rasterize the intersection to find which pixels are covered
            # We'll do fill=0 for outside, 1 for inside
            # (since polygon might be multipoly, union, etc.)
            from rasterio.features import rasterize

            inmask = rasterize(
                [(intersection_poly, 1)],
                out_shape=(H, W),
                fill=0,
                all_touched=True
            ).astype(bool)

            # Among these, find which are newly covered
            newly_covered = inmask & ~coverage_mask
            # Mark them as covered
            coverage_mask |= inmask

            # Record the level at which they got included
            pixel_levels[newly_covered] = lvl

            # gather them into a list of (r,c)
            new_pix = np.argwhere(newly_covered)
            # e.g. shape: (N,2), each row is [r,c]
            for rc in new_pix:
                newly_added_pixels.append((rc[0], rc[1]))

        # --- Summarize new coverage
        newly_count = len(newly_added_pixels)
        cumulative_area += newly_count

        # Check positivity among newly added
        newly_pos = [0] * n_channels
        for ch_idx in range(n_channels):
            img = images[ch_idx]
            if img is None:
                continue
            for (r, c) in newly_added_pixels:
                if img[r, c] > threshold:
                    newly_pos[ch_idx] += 1
            channel_cumul[ch_idx] += newly_pos[ch_idx]

        coverage_by_level[lvl] = {
            "area": cumulative_area,
            "positives": channel_cumul[:]
        }

    return coverage_by_level, pixel_levels


##############################################################################
# DEMO: CUMULATIVE AREA VS. BRANCH DEPTH PLOT
##############################################################################

def plot_cumulative_area_vs_depth(coverage_by_level, channel_labels=None):
    """
    Simple line plot of coverage area vs. branch depth, plus stacked lines
    of positives if desired.
    """
    import matplotlib.pyplot as plt

    # Sort levels
    levels = sorted(coverage_by_level.keys())
    area_vals = [coverage_by_level[l]["area"] for l in levels]
    positives_per_level = [coverage_by_level[l]["positives"] for l in levels]

    if channel_labels is None:
        n_channels = len(positives_per_level[0])
        channel_labels = [f"Channel {i}" for i in range(n_channels)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(levels, area_vals, '-ko', label="Cumulative Duct Area")

    # Overplot positives for each channel
    n_channels = len(positives_per_level[0])
    color_cycle = ["r", "g", "y", "c", "m", "b"]
    for ch_idx in range(n_channels):
        pvals = [p[ch_idx] for p in positives_per_level]
        ax.plot(levels, pvals,
                '-o', color=color_cycle[ch_idx % len(color_cycle)],
                label=channel_labels[ch_idx])

    ax.set_xlabel("Branch Depth (level)")
    ax.set_ylabel("Cumulative Pixel Count")
    ax.set_title("Coverage vs. Branch Depth")
    ax.legend()
    plt.tight_layout()
    plt.show()


##############################################################################
# EXAMPLE USAGE
##############################################################################
if __name__ == "__main__":

    from skimage import io
    from analysis.utils.loading_saving import load_duct_systems, create_directed_duct_graph
    import json

    json_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\hierarchy tree.json'
    duct_borders_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood.lif - TileScan 2 Merged_Processed001_outline1.geojson'

    green_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0001.tif'
    yellow_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0004.tif'
    red_image_path = r'I:\Group Rheenen\ExpDATA\2022_H.HRISTOVA\P004_TumorProgression_Myc\S005_Mouse_Puberty\E004_Imaging_3D\2473536_Cft_24W\25102024_2473536_R5_Ecad_sp8_maxgood-0006.tif'
    threshold_value = 1000
    system_idx = 1

    with open(duct_borders_path, 'r') as f:
        duct_borders = json.load(f)


    # Pseudo-code:
    # 1) Load the duct system and create a graph
    duct_systems = load_duct_systems(json_path)
    system_data = duct_systems[system_idx]
    G_dir = create_directed_duct_graph(system_data)

    # Find the root
    first_bp = list(G_dir.nodes)[0]
    while len(list(G_dir.predecessors(first_bp))) == 1:
        first_bp = list(G_dir.predecessors(first_bp))[0]
    root_node = first_bp
    print(f"Using branch point {root_node} as root")

    # 2) Load or build your duct_outline as a shapely geometry
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    # Suppose you read from a geojson:
    duct_outline = unary_union([
        make_valid(shape(feature['geometry']))
        for feature in duct_borders['features']
    ])

    # 3) Load images (e.g. red, green, yellow)
    # import skimage.io as io
    red_img = io.imread(red_image_path)
    green_img = io.imread(green_image_path)
    yellow_img = io.imread(yellow_image_path)
    images = [red_img, green_img, yellow_img]

    # 4) Run the coverage analysis
    coverage_by_level, pixel_levels = analyze_branch_depth_area(
        duct_system=system_data,
        G_dir=G_dir,
        root_node=root_node,
        duct_outline=duct_outline,
        images=images,
        threshold=500,
        buffer_width=5
    )

    # 5) Plot cumulative coverage
    plot_cumulative_area_vs_depth(coverage_by_level, channel_labels=["Red","Green","Yellow"])
    pass
