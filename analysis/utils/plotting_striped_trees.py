import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from analysis.utils.loading_saving import get_line_for_edge
from analysis.utils.plotting_trees import hierarchy_pos

def precompute_line_parameters(line):
    line_coords = np.array(line.coords)
    diffs = np.diff(line_coords, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative_lengths = np.concatenate(([0], np.cumsum(seg_lens)))
    total_length = cumulative_lengths[-1]
    return line_coords, diffs, seg_lens, cumulative_lengths, total_length


def project_pixels_onto_line(xs, ys, line_coords, segment_vecs, segment_lengths, cumulative_lengths):
    px = xs[:, None]
    py = ys[:, None]
    sx = line_coords[:-1, 0][None, :]
    sy = line_coords[:-1, 1][None, :]
    vx = segment_vecs[:, 0][None, :]
    vy = segment_vecs[:, 1][None, :]
    seg_len_sq = segment_lengths ** 2

    dot = (px - sx) * vx + (py - sy) * vy
    t = dot / seg_len_sq[None, :]
    t_clamped = np.clip(t, 0, 1)
    proj_x = sx + t_clamped * vx
    proj_y = sy + t_clamped * vy
    dx = px - proj_x
    dy = py - proj_y
    dist_sq = dx * dx + dy * dy
    idx_min = np.argmin(dist_sq, axis=1)
    t_chosen = t_clamped[np.arange(len(xs)), idx_min]
    chosen_seg_lengths = segment_lengths[idx_min]
    chosen_cum_lengths = cumulative_lengths[idx_min]
    fraction_dist = chosen_cum_lengths + t_chosen * chosen_seg_lengths
    min_dist_sq = dist_sq[np.arange(len(xs)), idx_min]
    return fraction_dist, min_dist_sq


def assign_pixels_to_subsegments(line, duct_mask, images, threshold, N=100, buffer_width=5):
    minx, miny, maxx, maxy = line.bounds
    minx = max(int(minx - buffer_width), 0)
    miny = max(int(miny - buffer_width), 0)
    maxx = min(int(maxx + buffer_width), duct_mask.shape[1] - 1)
    maxy = min(int(maxy + buffer_width), duct_mask.shape[0] - 1)
    sub_mask = duct_mask[miny:maxy + 1, minx:maxx + 1]
    ys, xs = np.where(sub_mask == 1)
    if len(xs) == 0:
        num_channels = len(images)
        return np.zeros((N, num_channels), dtype=int), np.zeros((N, num_channels), dtype=int)
    xs = xs + minx
    ys = ys + miny
    line_coords, segment_vecs, segment_lengths, cumulative_lengths, total_length = precompute_line_parameters(line)
    num_channels = len(images)
    if total_length == 0:
        return np.zeros((N, num_channels), dtype=int), np.zeros((N, num_channels), dtype=int)

    pixel_values = []
    for img in images:
        if img is None:
            pixel_values.append(np.zeros(len(xs), dtype=np.uint8))
        else:
            pixel_values.append(img[ys, xs])
    pixel_values = np.array(pixel_values).T
    fraction_dist, min_dist_sq = project_pixels_onto_line(xs, ys, line_coords, segment_vecs, segment_lengths,
                                                          cumulative_lengths)
    fractions = fraction_dist / total_length
    inside_corridor = (min_dist_sq <= buffer_width * buffer_width)
    if not np.any(inside_corridor):
        return np.zeros((N, num_channels), dtype=int), np.zeros((N, num_channels), dtype=int)
    fractions = fractions[inside_corridor]
    pixel_values = pixel_values[inside_corridor]
    bins = (fractions * N).astype(int)
    bins[bins == N] = N - 1
    totals = np.zeros((N, num_channels), dtype=int)
    positives = np.zeros((N, num_channels), dtype=int)
    for c in range(num_channels):
        channel_vals = pixel_values[:, c]
        is_positive = (channel_vals > threshold).astype(int)
        totals[:, c] = np.bincount(bins, minlength=N)
        positives[:, c] = np.bincount(bins, weights=is_positive, minlength=N)
    return totals, positives


def create_rgb_color_from_percentages(vals):
    scaled = [int((v / 100) * 255) for v in vals]  # [red, yellow, green]
    max_val = max(scaled)
    if max_val == 0:
        return '#000000'
    max_idx = scaled.index(max_val)
    if max_idx == 0:
        return '#ff0000'  # Red
    elif max_idx == 1:
        return '#00ff00'  # Green
    elif max_idx == 2:
        return '#ffff00'  # Yellow


def plot_hierarchical_graph_subsegments(
        G,
        root_node,
        duct_mask,
        red_image=None,
        green_image=None,
        yellow_image=None,
        threshold=500,
        N=100,
        draw_nodes=False,
        use_hierarchy_pos=False,
        vert_gap=1,
        orthogonal_edges=True,
        linewidth=1.5,
        buffer_width=5
):
    if not G.nodes:
        raise ValueError("The graph is empty.")
    pos = (hierarchy_pos(G, root=root_node, vert_gap=vert_gap) if use_hierarchy_pos
           else nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB', root=root_node))
    fig, ax = plt.subplots(figsize=(35, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    images = [red_image, green_image, yellow_image]
    segment_data_cache = {}

    for (u, v) in G.edges():
        # Use the edge's stored segment name (if any)
        segment_name = G[u][v].get('segment_name', f"{u}_to_{v}")
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # Get the line geometry for this edge directly from the graph.
        if segment_name not in segment_data_cache:
            line = get_line_for_edge(G, u, v)
            totals, positives = assign_pixels_to_subsegments(line, duct_mask, images,
                                                             threshold, N=N,
                                                             buffer_width=buffer_width)
            segment_data_cache[segment_name] = (totals, positives)
        totals, positives = segment_data_cache[segment_name]
        percentages = np.zeros_like(positives, dtype=float)
        mask_nonzero = (totals > 0)
        percentages[mask_nonzero] = (positives[mask_nonzero] / totals[mask_nonzero]) * 100

        if orthogonal_edges:
            horiz_len = abs(x2 - x1)
            vert_len = abs(y2 - y1)
            total_len = horiz_len + vert_len
            if y1 < y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                horiz_len = abs(x2 - x1)
                vert_len = abs(y2 - y1)
                total_len = horiz_len + vert_len
            if total_len > 0:
                corner_frac = horiz_len / total_len if total_len > 0 else 0
                sign = np.sign(y2 - y1)
                for i in range(N):
                    start_frac = i / N
                    end_frac = (i + 1) / N
                    dist_start = start_frac * total_len
                    dist_end = end_frac * total_len
                    vals = percentages[i, :]
                    color = create_rgb_color_from_percentages(vals)
                    if end_frac <= corner_frac:
                        sx = x1 + (x2 - x1) * (dist_start / horiz_len) if horiz_len > 0 else x1
                        sy = y1
                        ex = x1 + (x2 - x1) * (dist_end / horiz_len) if horiz_len > 0 else x1
                        ey = y1
                        ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                    elif start_frac >= corner_frac:
                        dist_start_vert = dist_start - horiz_len
                        dist_end_vert = dist_end - horiz_len
                        sx = x2
                        sy = y1 + sign * dist_start_vert if vert_len > 0 else y1
                        ex = x2
                        ey = y1 + sign * dist_end_vert if vert_len > 0 else y1
                        ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)
                    else:
                        dist_corner = horiz_len
                        sx = x1 + (x2 - x1) * (dist_start / horiz_len) if horiz_len > 0 else x1
                        sy = y1
                        cx = x1 + (x2 - x1) * (dist_corner / horiz_len) if horiz_len > 0 else x1
                        cy = y1
                        ax.plot([sx, cx], [sy, cy], color=color, linewidth=linewidth, zorder=1)
                        dist_start_vert = dist_corner - horiz_len
                        dist_end_vert = dist_end - horiz_len
                        vx = x2
                        vy_start = y1 + sign * dist_start_vert if vert_len > 0 else y1
                        vy_end = y1 + sign * dist_end_vert if vert_len > 0 else y1
                        ax.plot([vx, vx], [vy_start, vy_end], color=color, linewidth=linewidth, zorder=1)
        else:
            for i in range(N):
                start_frac = i / N
                end_frac = (i + 1) / N
                sx = x1 + (x2 - x1) * start_frac
                sy = y1 + (y2 - y1) * start_frac
                ex = x1 + (x2 - x1) * end_frac
                ey = y1 + (y2 - y1) * end_frac
                vals = percentages[i, :]
                color = create_rgb_color_from_percentages(vals)
                ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, zorder=1)

    if draw_nodes:
        for node in G.nodes():
            x, y = pos[node]
            ax.text(x, y, str(node), fontsize=5, ha='center', va='center', zorder=3, color='navy')

    plt.tight_layout()
    return fig, ax
