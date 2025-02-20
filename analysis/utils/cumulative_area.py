import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import deque
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from rasterio.features import rasterize


def compute_branch_levels(G_dir, root_node):
    """
    Returns a dict {node: level} indicating the branch level distance from `root_node`.
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
    An edge from u->v is grouped by the branch level of v.
    """
    level_dict = {}
    for u, v in G_dir.edges():
        seg_name = G_dir[u][v].get('segment_name', None)
        if seg_name is None:
            continue
        end_level = levels[v]
        level_dict.setdefault(end_level, []).append(seg_name)
    return level_dict


def get_line_for_segment(G, segment_name):
    """
    Given a graph G and a segment_name, find the edge with that segment_name
    and return its LineString. Assumes node attributes contain 'x' and 'y',
    and that edge attributes may contain 'internal_points'.
    """
    for u, v, data in G.edges(data=True):
        if data.get("segment_name") == segment_name:
            # Build the line using node positions
            start = G.nodes[u]
            end = G.nodes[v]
            pts = [(start["x"], start["y"])]
            if data.get("internal_points"):
                pts.extend([(p["x"], p["y"]) for p in data["internal_points"]])
            pts.append((end["x"], end["y"]))
            return LineString(pts)
    raise KeyError(f"Segment {segment_name} not found in graph.")

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
    # Buffer the line
    corridor_polygon = segment_line.buffer(buffer_dist)
    # Intersect with duct
    corridor_polygon = corridor_polygon.intersection(duct_polygon)
    if corridor_polygon.is_empty:
        return set(), {i: set() for i in range(len(images))}
    # Rasterize corridor
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
    channel_positive = {}
    for i, img in enumerate(images):
        if img is None:
            channel_positive[i] = set()
            continue
        pos_pix = {(row, col) for (row, col) in all_pixels if img[row, col] > threshold}
        channel_positive[i] = pos_pix
    return all_pixels, channel_positive


def analyze_area_vs_branch_level_multi_corridor(
    G_dir,
    root_node,
    duct_polygon,
    images,
    threshold=1000,
    buffer_dist=5.0
):
    """
    Perform branch-level corridor analysis:
      1) Compute branch levels from `root_node`.
      2) Group edges/segments by branch level.
      3) For each level:
         - Union all corridor polygons for segments at that level.
         - Rasterize to get a mask of new pixels.
         - Record branch level for new pixels.
         - Count new pixels that pass threshold in each channel.

    Returns:
      area_by_level: list of cumulative area (pixels) at each branch level.
      positives_by_level: {channel_idx: list of cumulative positives per level}
      pixel_levels: 2D array with branch level per pixel (0 => not included)
    """
    levels = compute_branch_levels(G_dir, root_node)
    if not levels:
        return [], {}, None

    max_level = max(levels.values())
    lvl_segments = segments_by_level(G_dir, levels)

    # Determine output shape from the first valid image.
    out_shape = None
    for img in images:
        if img is not None:
            out_shape = img.shape
            break
    if out_shape is None:
        raise ValueError("No valid image to determine out_shape.")

    area_by_level = [0]  # level 0 => 0 area
    num_channels = len(images)
    positives_by_level = {ch: [0] for ch in range(num_channels)}
    pixel_levels = np.zeros(out_shape, dtype=np.uint16)
    cumulative_mask = np.zeros(out_shape, dtype=bool)
    channel_cumulative_count = [0] * num_channels

    def rasterize_polygon(poly):
        return rasterize(
            [(poly, 1)],
            out_shape=out_shape,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        ).astype(bool)

    # Iterate over branch levels (starting at 1)
    for lvl in range(1, max_level + 1):
        seg_names = lvl_segments.get(lvl, [])
        if not seg_names:
            area_by_level.append(area_by_level[-1])
            for ch in range(num_channels):
                positives_by_level[ch].append(positives_by_level[ch][-1])
            continue

        corridors = []
        for seg in seg_names:
            try:
                line = get_line_for_segment(G_dir, seg)
            except KeyError:
                continue
            buff_poly = line.buffer(buffer_dist)
            corr_poly = buff_poly.intersection(duct_polygon)
            if not corr_poly.is_empty:
                corridors.append(corr_poly)
        union_poly = unary_union(corridors) if corridors else Polygon()

        if union_poly.is_empty:
            area_by_level.append(area_by_level[-1])
            for ch in range(num_channels):
                positives_by_level[ch].append(positives_by_level[ch][-1])
            continue

        corridor_mask_i = rasterize_polygon(union_poly)
        new_mask = corridor_mask_i & (~cumulative_mask)
        new_pixels_indices = np.where(new_mask)
        pixel_levels[new_pixels_indices] = lvl
        cumulative_mask |= corridor_mask_i
        new_pixels_count = np.count_nonzero(new_mask)
        current_area_total = area_by_level[-1] + new_pixels_count

        for ch in range(num_channels):
            if images[ch] is not None:
                ch_image = images[ch]
                above_thr_mask = (ch_image > threshold) & new_mask
                positives_count_new = np.count_nonzero(above_thr_mask)
                channel_cumulative_count[ch] += positives_count_new
            positives_by_level[ch].append(channel_cumulative_count[ch])
        area_by_level.append(current_area_total)

    return area_by_level, positives_by_level, pixel_levels

def plot_pixel_levels(pixel_levels, max_level):
    color_list = ["black"]
    cmap_viridis = plt.cm.get_cmap("viridis", max_level)
    for i in range(max_level):
        color_list.append(mpl.colors.rgb2hex(cmap_viridis(i)))
    discrete_cmap = mpl.colors.ListedColormap(color_list)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(pixel_levels, cmap=discrete_cmap, vmin=0, vmax=max_level)
    ax.set_title("Pixel Branch Level Map")
    ax.axis("off")
    ticks = range(max_level + 1)
    cbar = plt.colorbar(cax, ticks=ticks)
    cbar.set_ticks(ticks[::10])
    cbar.set_label("Branch Level")
    plt.tight_layout()
    return fig, ax

def plot_area_vs_branch_level_multi_stack(
    area_by_level,
    positives_by_level,
    channel_labels=None,
    use_log=False
):
    if channel_labels is None:
        channel_labels = [f"Channel {i}" for i in sorted(positives_by_level.keys())]
    xvals = range(len(area_by_level))
    sorted_channels = sorted(positives_by_level.keys())
    sum_of_positives = []
    for lvl in xvals:
        total_pos = sum(positives_by_level[ch][lvl] for ch in sorted_channels)
        sum_of_positives.append(total_pos)
    negative = []
    for i, total_area in enumerate(area_by_level):
        val = total_area - sum_of_positives[i]
        negative.append(val if val > 0 else 0)
    stacked_data = []
    for ch in sorted_channels:
        stacked_data.append(positives_by_level[ch])
    labels = [channel_labels[ch] for ch in sorted_channels]
    yellow = positives_by_level[2]
    green = positives_by_level[1]
    red = positives_by_level[0]
    print(f"Percentage of yellow pixels: {yellow[-1] / area_by_level[-1] * 100:.2f}%")
    print(f"Percentage of green pixels: {green[-1] / area_by_level[-1] * 100:.2f}%")
    print(f"Percentage of red pixels: {red[-1] / area_by_level[-1] * 100:.2f}%")
    print(f"Percentage of positive pixels: {(red[-1] + yellow[-1] + green[-1]) / area_by_level[-1] * 100:.2f}%")
    for i in range(len(stacked_data)):
        stacked_data[i] = [x / area_by_level[-1] for x in stacked_data[i]]
    colors = ["red", "green", "yellow"]
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
    return fig, ax